"""
core/bayesian.py
================
Байесовское обновление и UAF-расширенная схема весов свидетельств.

Формальное основание — THEORY.md §10:

    Стандартный Байес:
        P(M|E) = P(E|M) · P(M) / P(E)

    Log-odds:
        L_n = L_0 + Σ_i λ_i

    UAF-расширение (§10.3):
        λ_i^UAF = λ_i · R_i · I_i · S_i · F_i · (1 + G_i)

    где:
        R_i — надёжность источника (reliability)
        I_i — независимость свидетельства (independence)
        S_i — охват (scope)
        F_i — свежесть (freshness: e^{-λΔt})
        G_i — предиктивный выигрыш (predictive gain)

    P_UAF(M|E) = σ(L_UAF) = 1 / (1 + e^{-L_UAF})

    Матрица доверия ноосферы (§16.2):
        T_ij(t+1) = (1−β)·T_ij(t) + β·1[ŷ_j = y_actual]

Модуль независим. Нет зависимостей кроме стандартной библиотеки.
"""

import math
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any


# ============================================================
# БАЗОВЫЕ ФУНКЦИИ
# ============================================================

def sigmoid(x: float) -> float:
    """σ(x) = 1 / (1 + e^{-x})"""
    if x > 500:
        return 1.0
    if x < -500:
        return 0.0
    return 1.0 / (1.0 + math.exp(-x))


def log_odds(p: float, eps: float = 1e-9) -> float:
    """
    Логарифм отношения шансов: log[p / (1−p)].
    Монотонное отображение вероятности в (-∞, +∞).
    """
    p = max(eps, min(1 - eps, p))
    return math.log(p / (1.0 - p))


def from_log_odds(L: float) -> float:
    """Обратное: вероятность из логарифма шансов."""
    return sigmoid(L)


def normalize(d: Dict[str, float], eps: float = 1e-12) -> Dict[str, float]:
    s = sum(d.values())
    if s <= 0:
        n = len(d)
        return {k: 1.0 / n for k in d}
    return {k: max(v, eps) / s for k, v in d.items()}


def logsumexp(xs: List[float]) -> float:
    """log Σ exp(x_i) — численно стабильный."""
    m = max(xs)
    return m + math.log(sum(math.exp(x - m) for x in xs))


# ============================================================
# СВИДЕТЕЛЬСТВО
# ============================================================

@dataclass
class Evidence:
    """
    Одно свидетельство в UAF-системе.

    Поля соответствуют §10.3 THEORY.md.
    """
    # Содержание
    summary: str
    direction: float           # +1 за модель, -1 против, 0 нейтральное
    raw_strength: float        # |λ_i| — сила свидетельства

    # Качественные веса λ_UAF = λ · R · I · S · F · (1+G)
    reliability: float = 1.0   # R_i ∈ [0,1]: историческая надёжность источника
    independence: float = 1.0  # I_i ∈ [0,1]: независимость от других свидетельств
    scope: float = 1.0         # S_i ∈ [0,1]: область применимости
    freshness_halflife: float = 0.0  # τ для F_i = e^{-Δt/τ}; 0 → не стареет
    predictive_gain: float = 0.0  # G_i ≥ 0: дополнительный предиктивный выигрыш

    # Метаданные
    source: str = "unknown"
    timestamp: float = field(default_factory=time.time)

    @property
    def raw_lambda(self) -> float:
        """λ_i = direction × strength (до качественных весов)."""
        return self.direction * self.raw_strength

    def freshness(self, now: Optional[float] = None) -> float:
        """F_i = e^{-Δt / τ}. При τ=0 → F=1 (не стареет)."""
        if self.freshness_halflife <= 0:
            return 1.0
        dt = (now or time.time()) - self.timestamp
        return math.exp(-dt / self.freshness_halflife)

    def uaf_lambda(self, now: Optional[float] = None) -> float:
        """
        λ_i^UAF = λ_i · R_i · I_i · S_i · F_i · (1 + G_i)
        """
        return (
            self.raw_lambda
            * max(0.0, self.reliability)
            * max(0.0, self.independence)
            * max(0.0, self.scope)
            * self.freshness(now)
            * (1.0 + max(0.0, self.predictive_gain))
        )


# ============================================================
# БАЙЕСОВСКОЕ ОБНОВЛЕНИЕ
# ============================================================

class BayesianUpdater:
    """
    Стандартное байесовское обновление в log-odds пространстве.

    Использование:
        updater = BayesianUpdater(prior=0.5)
        updater.update(lambda_i=0.5)  # свидетельство за
        updater.update(lambda_i=-0.3) # свидетельство против
        print(updater.posterior)      # обновлённая вероятность
    """

    def __init__(self, prior: float = 0.5):
        self._L = log_odds(prior)
        self._history: List[Tuple[float, float]] = []  # (lambda, posterior)

    def update(self, lambda_i: float) -> float:
        """L_n = L_{n-1} + λ_i. Возвращает новый posterior."""
        self._L += lambda_i
        p = from_log_odds(self._L)
        self._history.append((lambda_i, p))
        return p

    @property
    def posterior(self) -> float:
        return from_log_odds(self._L)

    @property
    def log_odds_current(self) -> float:
        return self._L

    def reset(self, new_prior: float = 0.5):
        self._L = log_odds(new_prior)
        self._history.clear()


# ============================================================
# UAF UPDATER
# ============================================================

class UAFUpdater(BayesianUpdater):
    """
    Расширенное байесовское обновление с UAF-весами.

    Принимает объекты Evidence и автоматически вычисляет
    λ_UAF с учётом всех качественных множителей.

    Отличие от чистого Байеса:
        • Ненадёжные источники ослабляются (reliability)
        • Коррелированные свидетельства штрафуются (independence)
        • Устаревшие данные теряют вес (freshness)
        • Предсказательный успех источника усиливает вес (+G_i)
    """

    def __init__(self, prior: float = 0.5):
        super().__init__(prior)
        self._evidences: List[Evidence] = []

    def add_evidence(self, ev: Evidence) -> float:
        """
        Обновляет posterior по новому свидетельству.
        Возвращает новый posterior.
        """
        lam = ev.uaf_lambda()
        self._evidences.append(ev)
        return self.update(lam)

    def add_many(self, evidences: List[Evidence]) -> float:
        """Последовательно добавляет несколько свидетельств."""
        for ev in evidences:
            self.add_evidence(ev)
        return self.posterior

    @property
    def evidence_count(self) -> int:
        return len(self._evidences)

    def weight_summary(self) -> List[Dict[str, Any]]:
        """Отчёт о вкладе каждого свидетельства."""
        return [
            {
                "summary": ev.summary[:50],
                "raw_lambda": round(ev.raw_lambda, 4),
                "uaf_lambda": round(ev.uaf_lambda(), 4),
                "reliability": ev.reliability,
                "freshness": round(ev.freshness(), 3),
            }
            for ev in self._evidences
        ]


# ============================================================
# MULTI-HYPOTHESIS UPDATER
# ============================================================

class MultiHypothesisUpdater:
    """
    Байесовское обновление над множеством конкурирующих гипотез.

    Реализует §10 для случая нескольких интентов/моделей:
        P(H_i | E) ∝ P(E | H_i) · P(H_i)

    Пример:
        updater = MultiHypothesisUpdater({"math": 0.33, "code": 0.33, "theory": 0.34})
        updater.update("python bug function api")
        print(updater.belief)  # {"math": 0.03, "code": 0.94, "theory": 0.03}
    """

    def __init__(self, prior: Dict[str, float]):
        self.belief = normalize(prior)
        self._update_history: List[Dict[str, float]] = [dict(self.belief)]
        self._sources: Dict[str, BayesianUpdater] = {
            h: BayesianUpdater(p) for h, p in self.belief.items()
        }

    def update(
        self,
        log_likelihoods: Dict[str, float],
    ) -> Dict[str, float]:
        """
        P(H_i | E) ∝ exp(log P(H_i) + log P(E | H_i))

        Parameters
        ----------
        log_likelihoods : dict
            {hypothesis: log P(obs | hypothesis)}
        """
        log_posts = {
            h: math.log(max(self.belief[h], 1e-12)) + log_likelihoods.get(h, 0.0)
            for h in self.belief
        }
        m = max(log_posts.values())
        unnorm = {h: math.exp(v - m) for h, v in log_posts.items()}
        self.belief = normalize(unnorm)
        self._update_history.append(dict(self.belief))
        return self.belief

    def update_from_text(
        self,
        text: str,
        vocab: Dict[str, Dict[str, float]],
        clarify: bool = False,
    ) -> Dict[str, float]:
        """
        Обновление по тексту с заданным словарным профилем.

        vocab = {hypothesis: {token: probability}}
        """
        toks = text.lower().split()
        log_liks = {}
        for h, token_dist in vocab.items():
            unk = 1.0 / (20 * len(token_dist) + 1)
            ll = sum(math.log(token_dist.get(t, unk)) for t in toks) if toks else 0.0
            log_liks[h] = ll
        return self.update(log_liks)

    def top_hypothesis(self) -> Tuple[str, float]:
        """Возвращает (гипотеза с макс. вероятностью, вероятность)."""
        best = max(self.belief, key=self.belief.get)
        return best, self.belief[best]

    def entropy(self) -> float:
        """H(belief) — мера неопределённости."""
        eps = 1e-12
        return -sum(max(p, eps) * math.log(max(p, eps)) for p in self.belief.values())

    def is_certain(self, threshold: float = 0.85) -> bool:
        """True если один вариант доминирует."""
        _, p = self.top_hypothesis()
        return p >= threshold

    def divergence_from_prior(self, prior: Dict[str, float]) -> float:
        """D_KL(belief ‖ prior) — насколько belief отличается от начального."""
        eps = 1e-12
        return sum(
            max(self.belief.get(k, eps), eps)
            * math.log(max(self.belief.get(k, eps), eps) / max(prior.get(k, eps), eps))
            for k in set(self.belief) | set(prior)
        )


# ============================================================
# TRUST MATRIX (Ноосфера)
# ============================================================

class TrustMatrix:
    """
    Матрица доверия для мультиагентной системы.

    T_ij = доверие агента i к агенту j.

    Обновление (§16.2 THEORY.md):
        T_ij(t+1) = (1−β) · T_ij(t) + β · 1[ŷ_j = y_actual]

    Предел (Теорема 16.1):
        lim_{t→∞} T_ij = NPG_j / max_k NPG_k

    Авторитет — не социальный конструкт, а следствие
    предсказательного успеха.
    """

    def __init__(self, agent_ids: List[str], beta: float = 0.1, init_trust: float = 0.5):
        self.ids = agent_ids
        self.beta = beta
        self.n = len(agent_ids)
        self._idx = {a: i for i, a in enumerate(agent_ids)}
        self._T = [
            [0.0 if i == j else init_trust for j in range(self.n)]
            for i in range(self.n)
        ]
        self._npg: Dict[str, float] = {a: 0.0 for a in agent_ids}
        self._steps: Dict[str, int] = {a: 0 for a in agent_ids}

    def observe(self, observer: str, agent_j: str, prediction_correct: bool):
        """
        Агент observer наблюдает за результатом предсказания agent_j.
        T_ij(t+1) = (1−β)·T_ij(t) + β·1[correct]
        """
        i = self._idx[observer]
        j = self._idx[agent_j]
        self._T[i][j] = (1 - self.beta) * self._T[i][j] + self.beta * (1 if prediction_correct else 0)

    def update_npg(self, agent: str, step_pg: float):
        """Обновляет накопленный предиктивный выигрыш агента."""
        self._npg[agent] += step_pg
        self._steps[agent] += 1

    def trust(self, observer: str, agent: str) -> float:
        """T_ij ∈ [0,1]."""
        i, j = self._idx[observer], self._idx[agent]
        return self._T[i][j]

    def avg_trust_toward(self, agent: str) -> float:
        """Среднее доверие всех остальных агентов к данному."""
        j = self._idx[agent]
        others = [self._T[i][j] for i in range(self.n) if i != j]
        return sum(others) / max(len(others), 1)

    def theoretical_limit(self) -> Dict[str, float]:
        """
        Теоретический предел T_ij = NPG_j / max_k NPG_k.
        Показывает, к чему должно сойтись доверие.
        """
        avg_pgs = {a: self._npg[a] / max(self._steps[a], 1) for a in self.ids}
        max_pg = max(avg_pgs.values()) if avg_pgs.values() else 1.0
        if max_pg <= 0:
            return {a: 1.0 / self.n for a in self.ids}
        return {a: max(0.01, pg / max_pg) for a, pg in avg_pgs.items()}

    def is_converging_to_theory(self, tolerance: float = 0.15) -> bool:
        """
        Проверяет: сходится ли матрица доверия к теоретическому пределу?
        """
        theory = self.theoretical_limit()
        for agent in self.ids:
            avg_actual = self.avg_trust_toward(agent)
            theory_val = theory[agent]
            if abs(avg_actual - theory_val) > tolerance:
                return False
        return True

    def to_dict(self) -> Dict[str, Dict[str, float]]:
        return {
            self.ids[i]: {self.ids[j]: round(self._T[i][j], 3) for j in range(self.n)}
            for i in range(self.n)
        }


# ============================================================
# АНТИНОМИЯ ДЕТЕКТОР
# ============================================================

class AntinomyDetector:
    """
    Детектирует антиномии: ситуации, где P(T|E) ≈ P(¬T|E).

    §3 THEORY.md:
        Антиномия — не логический тупик, а сигнал о недостаточности
        свидетельств. Правильный ответ: «при текущих данных неразрешимо».

    Bayes Factor:
        BF = P(E | M_T) / P(E | M_{¬T})
        BF >> 1: T предпочтительнее
        BF ≈ 1:  неразрешимо (антиномия)
        BF << 1: ¬T предпочтительнее
    """

    def __init__(self, antinomy_threshold: float = 0.2):
        """
        antinomy_threshold : float
            |P(T) - P(¬T)| < threshold → считаем антиномией.
        """
        self.threshold = antinomy_threshold

    def check(self, posterior: float) -> Dict[str, Any]:
        """
        posterior : float
            P(T | E) — вероятность тезиса после всех свидетельств.

        Returns
        -------
        dict:
            is_antinomy : bool
            p_thesis : float
            p_antithesis : float
            bayes_factor : float
            verdict : str
        """
        p_t = posterior
        p_not_t = 1.0 - posterior
        bf = p_t / max(p_not_t, 1e-9)
        delta = abs(p_t - p_not_t)
        is_antinomy = delta < self.threshold

        if is_antinomy:
            verdict = "ANTINOMY: insufficient evidence (BF ≈ 1)"
        elif bf > 3:
            verdict = f"Thesis supported (BF={bf:.2f})"
        elif bf < 1 / 3:
            verdict = f"Antithesis supported (BF={bf:.2f})"
        else:
            verdict = f"Weak evidence (BF={bf:.2f})"

        return {
            "is_antinomy": is_antinomy,
            "p_thesis": round(p_t, 4),
            "p_antithesis": round(p_not_t, 4),
            "bayes_factor": round(bf, 4),
            "verdict": verdict,
        }


# ============================================================
# БЫСТРЫЕ ПРОВЕРКИ
# ============================================================

def _run_checks():
    print("bayesian.py checks...")

    # 1. Sigmoid / log_odds
    assert abs(sigmoid(0) - 0.5) < 1e-9
    assert abs(sigmoid(log_odds(0.7)) - 0.7) < 1e-6
    assert abs(sigmoid(100) - 1.0) < 1e-6
    print("  ✓ sigmoid / log_odds roundtrip")

    # 2. BayesianUpdater — монотонно растёт при положительных λ
    bu = BayesianUpdater(prior=0.5)
    p1 = bu.update(0.5)
    p2 = bu.update(0.5)
    assert p2 > p1 > 0.5
    print("  ✓ BayesianUpdater monotone with positive evidence")

    # 3. UAFUpdater — ненадёжный источник слабее
    ev_reliable = Evidence("reliable source", +1, 1.0, reliability=0.9, independence=0.9, scope=0.9)
    ev_unreliable = Evidence("unreliable source", +1, 1.0, reliability=0.1, independence=0.9, scope=0.9)
    assert ev_reliable.uaf_lambda() > ev_unreliable.uaf_lambda()
    print("  ✓ Reliable evidence has higher λ_UAF")

    # 4. UAFUpdater — устаревшее свидетельство слабее
    ev_fresh = Evidence("fresh", +1, 1.0, freshness_halflife=3600, timestamp=time.time())
    ev_stale = Evidence("stale", +1, 1.0, freshness_halflife=3600, timestamp=time.time() - 7200)
    assert ev_fresh.uaf_lambda() > ev_stale.uaf_lambda()
    print("  ✓ Fresh evidence has higher λ_UAF than stale")

    # 5. MultiHypothesisUpdater — код-токены сдвигают belief в code
    prior = {"math": 1/3, "code": 1/3, "theory": 1/3}
    mhu = MultiHypothesisUpdater(prior)
    vocab = {
        "math":   {"python": 0.01, "integral": 0.7, "api": 0.01},
        "code":   {"python": 0.6,  "integral": 0.01, "api": 0.5},
        "theory": {"python": 0.02, "integral": 0.05, "api": 0.05},
    }
    mhu.update_from_text("python api function", vocab)
    assert mhu.belief["code"] > 0.7
    print(f"  ✓ MultiHypothesisUpdater: code={mhu.belief['code']:.3f}")

    # 6. TrustMatrix обновляется корректно
    tm = TrustMatrix(["A", "B", "C"], beta=0.2)
    # B всегда прав
    for _ in range(20):
        tm.observe("A", "B", True)
        tm.observe("C", "B", True)
        tm.update_npg("B", 0.5)
    # C всегда ошибается
    for _ in range(20):
        tm.observe("A", "C", False)
        tm.observe("B", "C", False)
        tm.update_npg("C", -0.5)
    assert tm.avg_trust_toward("B") > tm.avg_trust_toward("C")
    print(f"  ✓ TrustMatrix: trust(B)={tm.avg_trust_toward('B'):.3f} > trust(C)={tm.avg_trust_toward('C'):.3f}")

    # 7. AntinomyDetector
    ad = AntinomyDetector(antinomy_threshold=0.2)
    result_antinomy = ad.check(0.52)
    result_clear = ad.check(0.85)
    assert result_antinomy["is_antinomy"]
    assert not result_clear["is_antinomy"]
    print(f"  ✓ AntinomyDetector: p=0.52 → antinomy, p=0.85 → clear")

    print("\nAll bayesian.py checks passed. ✓")


if __name__ == "__main__":
    _run_checks()
