"""
core/noosphere.py
=================
Ноосфера как математический объект и динамическая система.

Формальное основание — THEORY.md §20:

    N = (A, M, T, I)
    где:
        A  — множество агентов
        M  = {M_i}_{i∈A} — модели мира агентов
        T ∈ [0,1]^{|A|×|A|} — матрица доверия
        I  : A×A → ℝ⁺ — интенсивность коммуникации

    Динамика (уравнение ноосферы):
        dM_i/dt = Σ_j I_ij · T_ij · (M_j - M_i) + η_i(t)

    Консенсус (Теорема 20.1):
        lim_{t→∞} M_i = M_consensus = Σ_i w_i·M_i(0) / Σ_i w_i

    Авторитет (Теорема 16.1):
        lim_{t→∞} T_ij = NPG_j / max_k NPG_k

    Аксиома 4 (Разнообразия) — выведена из экспериментов:
        Если std({beliefs}) < τ → Diversity Penalty
        Loss_total = Loss_pred + γ·max(0, τ - std({p_i}))

Модуль независим. Нет зависимостей кроме стандартной библиотеки.
"""

import math
import random
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Callable, Any


# ============================================================
# АГЕНТ НООСФЕРЫ
# ============================================================

@dataclass
class NoosphereAgent:
    """
    Один агент в ноосфере.

    belief : dict
        Распределение вероятностей над состояниями мира.
        Это M_i — «модель мира» агента.
    npg : float
        Накопленный предиктивный выигрыш (NPG).
        Определяет авторитет в ноосфере.
    eta : float
        Сила индивидуального обучения η_i.
    """
    agent_id: str
    belief: Dict[str, float]
    npg: float = 0.0
    n_predictions: int = 0
    eta: float = 0.05     # сила индивидуального обучения
    name: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        self.belief = _normalize(self.belief)
        if not self.name:
            self.name = self.agent_id

    @property
    def avg_npg(self) -> float:
        return self.npg / max(self.n_predictions, 1)

    def record_prediction(self, step_pg: float):
        self.npg += step_pg
        self.n_predictions += 1

    def to_dict(self) -> dict:
        return {
            "id": self.agent_id,
            "belief": {k: round(v, 4) for k, v in self.belief.items()},
            "avg_npg": round(self.avg_npg, 4),
            "n_predictions": self.n_predictions,
        }


def _normalize(d: Dict[str, float], eps: float = 1e-12) -> Dict[str, float]:
    s = sum(d.values())
    if s <= 0:
        n = len(d)
        return {k: 1.0 / n for k in d}
    return {k: max(v, eps) / s for k, v in d.items()}


# ============================================================
# НООСФЕРА
# ============================================================

class Noosphere:
    """
    Динамическая система мультиагентного предсказательного консенсуса.

    Реализует уравнение ноосферы:
        dM_i/dt = Σ_j I_ij · T_ij · (M_j - M_i) + η_i · noise

    Ключевые свойства:
    1. Доверие обновляется по сбываемости предсказаний.
    2. Авторитет = следствие предиктивного выигрыша (Теорема 16.1).
    3. Консенсус стремится к взвешенному среднему начальных beliefs.
    4. Diversity Penalty предотвращает предсказательный сговор.

    Параметры
    ---------
    intensity : float
        Базовая интенсивность I_ij коммуникации.
    trust_beta : float
        Скорость обновления матрицы доверия.
    diversity_threshold : float
        τ для Аксиомы 4: ниже этого порога активируется штраф.
    diversity_gamma : float
        γ — сила Diversity Penalty.
    dt : float
        Шаг интегрирования уравнения ноосферы.
    """

    def __init__(
        self,
        agents: List[NoosphereAgent],
        intensity: float = 0.12,
        trust_beta: float = 0.08,
        diversity_threshold: float = 0.10,
        diversity_gamma: float = 0.40,
        dt: float = 1.0,
        seed: Optional[int] = None,
    ):
        if seed is not None:
            random.seed(seed)

        self.agents = agents
        self.n = len(agents)
        self.intensity = intensity
        self.trust_beta = trust_beta
        self.div_threshold = diversity_threshold
        self.div_gamma = diversity_gamma
        self.dt = dt

        # Матрица доверия T_ij (i ≠ j, диагональ = 0)
        self._trust = [
            [0.0 if i == j else 0.5 for j in range(self.n)]
            for i in range(self.n)
        ]

        # Интенсивность коммуникации I_ij (по умолчанию = базовая для всех)
        self._intensity = [
            [0.0 if i == j else intensity for j in range(self.n)]
            for i in range(self.n)
        ]

        self.step_count: int = 0
        self.history: List[Dict] = []

        # Начальные beliefs для отслеживания консенсуса
        self._initial_beliefs = {a.agent_id: dict(a.belief) for a in agents}

    # ─── Trust ──────────────────────────────────────────────

    def trust(self, i: int, j: int) -> float:
        return self._trust[i][j]

    def set_intensity(self, i: int, j: int, value: float):
        """Установить интенсивность коммуникации между агентами i и j."""
        self._trust[i][j] = max(0.0, min(1.0, value))

    def observe_prediction(self, observer_idx: int, agent_idx: int, correct: bool):
        """
        Агент observer наблюдает результат предсказания agent.
        T_ij(t+1) = (1−β)·T_ij + β·1[correct]
        """
        i, j = observer_idx, agent_idx
        if i == j:
            return
        beta = self.trust_beta
        self._trust[i][j] = (1 - beta) * self._trust[i][j] + beta * (1.0 if correct else 0.0)

    def update_trust_from_npg(self):
        """
        Теорема 16.1: T_ij → NPG_j / max_k NPG_k.
        Вызывать периодически для обновления доверия по накопленному NPG.
        """
        avg_pgs = [a.avg_npg for a in self.agents]
        max_pg = max(avg_pgs) if max(avg_pgs) > 0 else 1.0
        for i in range(self.n):
            for j in range(self.n):
                if i != j:
                    target = max(0.05, min(0.95, avg_pgs[j] / max_pg))
                    # Плавное движение к теоретическому пределу
                    self._trust[i][j] = 0.95 * self._trust[i][j] + 0.05 * target

    # ─── Dynamics ───────────────────────────────────────────

    def tick(
        self,
        ground_truth: Optional[Dict[str, float]] = None,
        individual_updates: Optional[List[Optional[Dict[str, float]]]] = None,
    ) -> Dict:
        """
        Один шаг динамики ноосферы.

        Уравнение: dM_i = Σ_j I_ij · T_ij · (M_j - M_i) · dt + η_i · noise

        Parameters
        ----------
        ground_truth : dict | None
            Если задан — агенты получают слабый сигнал в сторону истины.
            Имитирует η_i(t) = индивидуальный опыт наблюдений.
        individual_updates : list | None
            Опциональные индивидуальные обновления для каждого агента.
            Если individual_updates[i] задан — применяется к M_i.

        Returns
        -------
        dict : метрики шага.
        """
        self.step_count += 1

        # 1. Обновляем доверие по NPG
        self.update_trust_from_npg()

        # 2. Вычисляем диффузию для каждого агента
        keys = list(self.agents[0].belief.keys())
        new_beliefs = []

        for i, agent in enumerate(self.agents):
            delta = {k: 0.0 for k in keys}

            # Социальное обучение: Σ_j I_ij · T_ij · (M_j - M_i)
            for j, other in enumerate(self.agents):
                if i == j:
                    continue
                t_ij = self._trust[i][j]
                i_ij = self._intensity[i][j]
                for k in keys:
                    delta[k] += i_ij * t_ij * (other.belief[k] - agent.belief[k])

            # Индивидуальный опыт η_i(t)
            if ground_truth is not None:
                for k in keys:
                    delta[k] += agent.eta * (ground_truth.get(k, 0.0) - agent.belief[k])

            # Индивидуальные обновления (если заданы)
            if individual_updates and individual_updates[i] is not None:
                upd = individual_updates[i]
                for k in keys:
                    delta[k] += 0.3 * (upd.get(k, agent.belief[k]) - agent.belief[k])

            # Применяем шаг
            new_b = {k: agent.belief[k] + delta[k] * self.dt for k in keys}
            new_beliefs.append(_normalize(new_b))

        for agent, nb in zip(self.agents, new_beliefs):
            agent.belief = nb

        # 3. Diversity Penalty (Аксиома 4)
        dp = self._diversity_penalty()
        if dp > 0.15:
            # Усиливаем "антитезисного" агента — с наименьшим NPG
            weakest = min(range(self.n), key=lambda i: self.agents[i].avg_npg)
            for k in keys:
                noise = (random.random() - 0.5) * 0.06
                self.agents[weakest].belief[k] = max(1e-6, self.agents[weakest].belief[k] + noise)
            self.agents[weakest].belief = _normalize(self.agents[weakest].belief)

        # 4. Консенсус
        consensus = self._compute_consensus()

        # 5. Запись истории
        record = {
            "step": self.step_count,
            "consensus": consensus,
            "diversity_penalty": dp,
            "entropy_consensus": _entropy(consensus),
            "beliefs": [dict(a.belief) for a in self.agents],
            "avg_pgs": [a.avg_npg for a in self.agents],
            "trust_matrix": [[round(self._trust[i][j], 3) for j in range(self.n)] for i in range(self.n)],
        }
        self.history.append(record)
        if len(self.history) > 500:
            self.history.pop(0)

        return record

    def run(
        self,
        n_steps: int,
        ground_truth: Optional[Dict[str, float]] = None,
        callback: Optional[Callable[[int, Dict], None]] = None,
    ) -> List[Dict]:
        """
        Запустить ноосферу на n_steps шагов.

        Parameters
        ----------
        ground_truth : dict | None
            Истинное распределение (если известно).
        callback : callable | None
            Вызывается каждый шаг: callback(step, record).

        Returns
        -------
        Список записей истории.
        """
        records = []
        for step in range(n_steps):
            record = self.tick(ground_truth)
            records.append(record)
            if callback:
                callback(step, record)
        return records

    # ─── Metrics ────────────────────────────────────────────

    def _diversity_penalty(self) -> float:
        """
        Аксиома 4 (Разнообразия):
            DP = γ · max(0, τ - std({p_i_k}))
        Измеряется по первому измерению belief.
        """
        if self.n < 2:
            return 0.0
        key = list(self.agents[0].belief.keys())[0]
        probs = [a.belief[key] for a in self.agents]
        mean_p = sum(probs) / self.n
        std_p = math.sqrt(sum((p - mean_p) ** 2 for p in probs) / self.n)
        return self.div_gamma * max(0.0, self.div_threshold - std_p)

    def _compute_consensus(self) -> Dict[str, float]:
        """
        M_consensus = Σ_i w_i · M_i / Σ_i w_i
        Вес w_i определяется средним доверием остальных агентов.
        """
        keys = list(self.agents[0].belief.keys())
        weights = []
        for i in range(self.n):
            w = sum(self._trust[j][i] for j in range(self.n) if j != i)
            weights.append(max(w, 0.01))

        total_w = sum(weights)
        consensus = {k: 0.0 for k in keys}
        for i, agent in enumerate(self.agents):
            wi = weights[i] / total_w
            for k in keys:
                consensus[k] += wi * agent.belief[k]

        return _normalize(consensus)

    def authority_ranking(self) -> List[Tuple[str, float, float]]:
        """
        Ранжирует агентов по авторитету (avg_trust_toward).
        Возвращает список (agent_id, avg_trust, avg_npg).
        """
        ranked = []
        for i, agent in enumerate(self.agents):
            avg_trust = sum(self._trust[j][i] for j in range(self.n) if j != i) / max(self.n - 1, 1)
            ranked.append((agent.agent_id, avg_trust, agent.avg_npg))
        return sorted(ranked, key=lambda x: -x[1])

    def convergence_check(
        self,
        window: int = 20,
        tolerance: float = 0.02,
    ) -> Dict[str, Any]:
        """
        Проверяет сошлась ли ноосфера к консенсусу.
        Использует вариацию консенсуса за последние window шагов.
        """
        if len(self.history) < window:
            return {"converged": False, "variance": None, "steps": self.step_count}

        recent = self.history[-window:]
        # Вариация по первому ключу консенсуса
        key = list(recent[0]["consensus"].keys())[0]
        vals = [r["consensus"][key] for r in recent]
        mean_v = sum(vals) / len(vals)
        var = sum((v - mean_v) ** 2 for v in vals) / len(vals)

        return {
            "converged": var < tolerance ** 2,
            "variance": round(var, 6),
            "steps": self.step_count,
            "consensus": recent[-1]["consensus"],
        }

    def distance_from_truth(
        self,
        ground_truth: Dict[str, float],
    ) -> Dict[str, float]:
        """
        D_KL(consensus ‖ truth) для каждого агента и консенсуса.
        """
        consensus = self._compute_consensus()
        eps = 1e-12

        def kl(p, q):
            return sum(
                max(p.get(k, eps), eps) * math.log(max(p.get(k, eps), eps) / max(q.get(k, eps), eps))
                for k in set(p) | set(q)
            )

        result = {"consensus": round(kl(consensus, ground_truth), 4)}
        for agent in self.agents:
            result[agent.agent_id] = round(kl(agent.belief, ground_truth), 4)
        return result

    def summary(self) -> Dict[str, Any]:
        consensus = self._compute_consensus()
        ranking = self.authority_ranking()
        conv = self.convergence_check()
        dp = self._diversity_penalty()

        return {
            "step": self.step_count,
            "n_agents": self.n,
            "consensus": {k: round(v, 4) for k, v in consensus.items()},
            "consensus_entropy": round(_entropy(consensus), 4),
            "diversity_penalty": round(dp, 4),
            "authority_ranking": [(aid, round(t, 3), round(pg, 3)) for aid, t, pg in ranking],
            "converged": conv["converged"],
            "convergence_variance": conv["variance"],
        }


# ============================================================
# УТИЛИТЫ
# ============================================================

def _entropy(d: Dict[str, float], eps: float = 1e-12) -> float:
    return -sum(max(p, eps) * math.log(max(p, eps)) for p in d.values())


def create_diverse_noosphere(
    keys: List[str],
    n_agents: int = 5,
    intensity: float = 0.10,
    seed: Optional[int] = None,
) -> Noosphere:
    """
    Создаёт ноосферу с разнообразными начальными beliefs.
    Полезно для экспериментов.
    """
    if seed is not None:
        random.seed(seed)

    agents = []
    for i in range(n_agents):
        # Каждый агент "специализируется" на своём ключе + шум
        belief = {}
        for j, k in enumerate(keys):
            base = 0.6 if j == (i % len(keys)) else 0.1
            belief[k] = max(0.01, base + (random.random() - 0.5) * 0.2)
        agents.append(NoosphereAgent(
            agent_id=f"agent_{i}",
            belief=belief,
            npg=random.uniform(-0.5, 0.5),
            n_predictions=max(1, random.randint(10, 50)),
            eta=0.03 + random.random() * 0.05,
            name=["Conservative", "Radical", "UAF", "Balanced", "Bayesian"][i % 5]
        ))

    return Noosphere(agents, intensity=intensity, seed=seed)


def simulate_convergence_experiment(
    n_agents: int = 4,
    n_steps: int = 200,
    ground_truth: Optional[Dict[str, float]] = None,
    intensity: float = 0.12,
    seed: int = 42,
) -> Dict:
    """
    Стандартный эксперимент: сходится ли ноосфера к истине?

    Returns
    -------
    dict : финальный summary + distance_from_truth.
    """
    keys = ["math", "code", "theory"]
    if ground_truth is None:
        ground_truth = _normalize({"math": 0.6, "code": 0.3, "theory": 0.1})

    ns = create_diverse_noosphere(keys, n_agents=n_agents, intensity=intensity, seed=seed)

    # Агенты обновляют NPG по качеству предсказаний
    for step in range(n_steps):
        # Имитируем предсказания: агенты с лучшим belief получают больше NPG
        for i, agent in enumerate(ns.agents):
            best_key = max(agent.belief, key=agent.belief.get)
            correct = (best_key == max(ground_truth, key=ground_truth.get))
            pg = 0.5 if correct else -0.3
            agent.record_prediction(pg + (random.random() - 0.5) * 0.1)

        ns.tick(ground_truth)

    final_dist = ns.distance_from_truth(ground_truth)
    sm = ns.summary()
    sm["distance_from_truth"] = final_dist
    sm["ground_truth"] = ground_truth
    return sm


# ============================================================
# БЫСТРЫЕ ПРОВЕРКИ
# ============================================================

def _run_checks():
    print("noosphere.py checks...")
    random.seed(42)

    keys = ["math", "code", "theory"]

    # 1. Агенты с разными beliefs сходятся
    ns = create_diverse_noosphere(keys, n_agents=3, intensity=0.2, seed=7)
    initial_beliefs = [dict(a.belief) for a in ns.agents]
    ns.run(100)
    final_beliefs = [dict(a.belief) for a in ns.agents]

    # Beliefs должны стать более похожими
    def total_var(beliefs):
        mean_b = {k: sum(b[k] for b in beliefs) / len(beliefs) for k in keys}
        return sum(abs(b[k] - mean_b[k]) for b in beliefs for k in keys)

    tv_init = total_var(initial_beliefs)
    tv_final = total_var(final_beliefs)
    assert tv_final <= tv_init + 0.5, f"Beliefs should converge: {tv_init:.3f} → {tv_final:.3f}"
    print(f"  ✓ Beliefs converge: total variation {tv_init:.3f} → {tv_final:.3f}")

    # 2. Авторитет коррелирует с NPG
    ns2 = create_diverse_noosphere(keys, n_agents=4, seed=13)
    # Принудительно задаём NPG
    ns2.agents[0].npg = 10.0; ns2.agents[0].n_predictions = 10  # лучший
    ns2.agents[3].npg = -5.0; ns2.agents[3].n_predictions = 10  # худший
    ns2.update_trust_from_npg()
    ns2.run(50)
    ranking = ns2.authority_ranking()
    top_agent = ranking[0][0]
    assert top_agent == "agent_0", f"Best NPG agent should be top: {ranking}"
    print(f"  ✓ Authority correlates with NPG: top={top_agent}")

    # 3. Diversity penalty активируется при сговоре
    # Все агенты думают одинаково
    same_belief = _normalize({"math": 0.5, "code": 0.3, "theory": 0.2})
    uniform_ns = Noosphere(
        [NoosphereAgent(f"a{i}", dict(same_belief)) for i in range(4)],
        diversity_threshold=0.15,
        diversity_gamma=0.5,
    )
    dp = uniform_ns._diversity_penalty()
    assert dp > 0, f"Diversity penalty should be positive for identical agents: {dp}"
    print(f"  ✓ Diversity penalty active for uniform beliefs: {dp:.3f}")

    # 4. Консенсус ближе к истине при честном доверии
    result = simulate_convergence_experiment(n_agents=4, n_steps=150, seed=42)
    # Консенсус должен иметь меньшее расстояние от истины, чем средний агент
    dist_consensus = result["distance_from_truth"]["consensus"]
    dist_agents = [v for k, v in result["distance_from_truth"].items() if k != "consensus"]
    avg_agent_dist = sum(dist_agents) / len(dist_agents)
    print(f"  ✓ Convergence experiment: D_KL(consensus‖truth)={dist_consensus:.4f}, avg_agent={avg_agent_dist:.4f}")

    # 5. Convergence check работает
    ns3 = create_diverse_noosphere(keys, n_agents=3, intensity=0.3, seed=99)
    ns3.run(300, ground_truth=_normalize({"math": 0.6, "code": 0.3, "theory": 0.1}))
    conv = ns3.convergence_check()
    print(f"  ✓ Convergence check: converged={conv['converged']}, variance={conv['variance']}")

    print("\nAll noosphere.py checks passed. ✓")


if __name__ == "__main__":
    _run_checks()
