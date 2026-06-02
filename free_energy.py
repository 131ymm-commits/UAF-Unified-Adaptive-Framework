"""
core/free_energy.py
===================
Реализация принципа свободной энергии Фристона.

Формальное основание — THEORY.md §2:

    F  = D_KL[Q(w) ‖ P(w|E)] − log P(E)
    F  = U − T·H                           (рабочая форма)
    U  = 𝔼[−log P(w_{t+1} | w_{≤t})]     (энергия = удивление)
    H  = −Σ P(x) log P(x)                 (энтропия модели)
    T  ∈ (0, ∞)                           (температура)

Три эквивалентных интерпретации минимизации F:
    • learning  — улучшить Q так, чтобы Q ≈ P(w|E)
    • action    — изменить E так, чтобы P(E) стал больше
    • attention — выбрать E с максимальной информационной ценностью

Модуль независим от остальной системы. Импортируется куда угодно.
"""

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence


# ============================================================
# БАЗОВЫЕ КОМПОНЕНТЫ
# ============================================================

def entropy(distribution: Dict[str, float], eps: float = 1e-12) -> float:
    """
    Шенноновская энтропия.
    H(X) = −Σ P(x) log P(x)

    H = 0: распределение вырожденное (полная уверенность)
    H = log|X|: равномерное распределение (полная неопределённость)
    """
    return -sum(
        max(p, eps) * math.log(max(p, eps))
        for p in distribution.values()
    )


def normalized_entropy(distribution: Dict[str, float]) -> float:
    """
    Нормированная энтропия H / log|X| ∈ [0, 1].
    0 = полная уверенность, 1 = максимальная неопределённость.
    """
    n = len(distribution)
    if n <= 1:
        return 0.0
    H = entropy(distribution)
    return H / math.log(n)


def surprise(prob: float) -> float:
    """
    Удивление (surprisal) конкретного события.
    S(x) = −log P(x)

    Чем реже событие, тем больше удивление.
    Среднее удивление = энтропия.
    """
    return -math.log(max(prob, 1e-12))


def kl_divergence(
    p: Dict[str, float],
    q: Dict[str, float],
    eps: float = 1e-12,
) -> float:
    """
    Дивергенция Кульбака–Лейблера D_KL(P‖Q).

    D_KL(P‖Q) = Σ P(x) log [P(x) / Q(x)] ≥ 0

    Свойство: D_KL(P‖Q) ≠ D_KL(Q‖P) (несимметрична).
    Равна нулю тогда и только тогда, когда P = Q.

    Применение в UAF (§8 THEORY.md):
        D_KL(M_0‖M) = NPG (предиктивный выигрыш)
        D_KL(P_t‖P_true) = функция Ляпунова познания
    """
    keys = set(p) | set(q)
    return sum(
        p.get(k, eps) * math.log(p.get(k, eps) / max(q.get(k, eps), eps))
        for k in keys
    )


def mutual_information(
    joint: Dict[tuple, float],
    marginal_x: Dict[str, float],
    marginal_y: Dict[str, float],
    eps: float = 1e-12,
) -> float:
    """
    Взаимная информация I(X;Y) = H(X) + H(Y) - H(X,Y).

    В THEORY.md §7: I(W;M) → H(W) — условие абсолютного знания.
    Мера «приближения к абсолютному знанию» = I(W;M) / H(W).
    """
    hx = entropy(marginal_x, eps)
    hy = entropy(marginal_y, eps)
    hxy = -sum(
        max(p, eps) * math.log(max(p, eps))
        for p in joint.values()
    )
    return hx + hy - hxy


# ============================================================
# СВОБОДНАЯ ЭНЕРГИЯ
# ============================================================

@dataclass
class FreeEnergyState:
    """Состояние системы с точки зрения свободной энергии."""
    U: float = 1.0        # Энергия (среднее удивление)
    T: float = 1.0        # Температура (мера накопленной ошибки)
    H: float = 0.5        # Энтропия модели (нормированная)
    F: float = 0.5        # Свободная энергия F = U − T·H
    residuum: float = 0.2 # Накопленный резидуум (EMA удивлений)
    step: int = 0


def compute_free_energy(U: float, T: float, H: float) -> float:
    """
    F = U − T·H

    U — энергия (удивление): чем хуже предсказание, тем выше.
    T — температура: растёт с накопленной ошибкой.
    H — энтропия: мера неопределённости модели.

    F < 0: модель хорошо предсказывает И имеет некоторую неопределённость.
    F > 0: удивление доминирует над неопределённостью.

    Минимизация F = баланс точности и неопределённости.
    """
    return U - T * H


class FreeEnergyTracker:
    """
    Отслеживает F = U − T·H по шагам обучения.

    Параметры
    ---------
    residuum_alpha : float
        Коэффициент EMA для накопления резидуума.
        Близко к 1 → медленная адаптация (долгая память).
        Близко к 0 → быстрая адаптация (короткая память).
    temp_min, temp_max : float
        Ограничения на температуру T.
    history_len : int
        Максимальная длина истории.
    """

    def __init__(
        self,
        residuum_alpha: float = 0.96,
        temp_min: float = 0.2,
        temp_max: float = 3.0,
        history_len: int = 500,
    ):
        self.alpha = residuum_alpha
        self.temp_min = temp_min
        self.temp_max = temp_max
        self.history_len = history_len

        self.state = FreeEnergyState()
        self.history: List[FreeEnergyState] = []

    def update(
        self,
        surprisal: float,
        model_distribution: Dict[str, float],
    ) -> FreeEnergyState:
        """
        Обновление состояния по новому наблюдению.

        Parameters
        ----------
        surprisal : float
            Удивление текущего шага: −log P(observation).
        model_distribution : dict
            Текущее распределение вероятностей модели.
            Используется для вычисления энтропии H.

        Returns
        -------
        FreeEnergyState : обновлённое состояние.
        """
        s = self.state
        s.step += 1

        # Накопление резидуума (EMA)
        s.residuum = self.alpha * s.residuum + (1 - self.alpha) * surprisal

        # Температура T: растёт с накопленной ошибкой
        s.T = max(self.temp_min, min(self.temp_max, 0.5 + s.residuum * 0.8))

        # Энергия U: текущее удивление (обрезаем снизу)
        s.U = max(0.0, min(surprisal, 10.0))

        # Энтропия H: нормированная
        s.H = normalized_entropy(model_distribution) if model_distribution else 0.5

        # Свободная энергия
        s.F = compute_free_energy(s.U, s.T, s.H)

        # Сохраняем снимок
        self._save_snapshot()
        return s

    def _save_snapshot(self):
        from copy import copy
        snap = copy(self.state)
        self.history.append(snap)
        if len(self.history) > self.history_len:
            self.history.pop(0)

    def convergence_rate(self) -> Optional[float]:
        """
        ρ = limsup V(t+1) / V(t) — скорость сходимости (§12 THEORY.md).

        ρ < 1: экспоненциальная сходимость
        ρ = 1: сублинейная сходимость
        ρ > 1: дивергенция (кризис парадигмы)
        """
        if len(self.history) < 10:
            return None
        recent = [s.F for s in self.history[-20:]]
        ratios = [
            abs(recent[i+1]) / max(abs(recent[i]), 1e-6)
            for i in range(len(recent) - 1)
            if abs(recent[i]) > 1e-6
        ]
        return max(ratios) if ratios else None

    def is_diverging(self, threshold: float = 1.2) -> bool:
        """Возвращает True если система дивергирует (кризис парадигмы)."""
        rho = self.convergence_rate()
        return rho is not None and rho > threshold

    def summary(self) -> Dict[str, float]:
        s = self.state
        return {
            "F": round(s.F, 4),
            "U": round(s.U, 4),
            "T": round(s.T, 4),
            "H": round(s.H, 4),
            "residuum": round(s.residuum, 4),
            "step": s.step,
        }


# ============================================================
# ПРЕДИКТИВНЫЙ ВЫИГРЫШ
# ============================================================

class PredictiveGainTracker:
    """
    Отслеживает накопленный предиктивный выигрыш NPG.

    NPG(M, T) = Σ_t log [M(w_t | w_{<t}) / M_0(w_t | w_{<t})]

    Базовая модель M_0:
        - 'uniform': равномерное распределение по словарю
        - 'unigram': унигрaмная статистика
        - кастомная: задаётся через baseline_logprob

    UAF-шкала: scale = tanh(NPG / steps / 2) × 3 ∈ [-3, +3]

    THEORY.md §9 Теорема 9.1:
        M* = lim_{T→∞} argmax_M (1/T) NPG(M, T)
        — абсолютная истина как предел максимального выигрыша.
    """

    SCALE_LABELS = {
        3: "уникальное предсказание",
        2: "лучше всех доступных",
        1: "лучше базовой",
        0: "как случайность",
        -1: "хуже базовой",
        -2: "стабильное анти-предсказание",
        -3: "оптимизирован на ошибки",
    }

    def __init__(self, vocab_size: int = 1000, baseline: str = "uniform"):
        self.vocab_size = vocab_size
        self.baseline = baseline
        self._log_baseline = -math.log(vocab_size) if baseline == "uniform" else 0.0
        self.npg: float = 0.0
        self.n_steps: int = 0
        self.step_gains: List[float] = []

    def update(
        self,
        model_logprob: float,
        baseline_logprob: Optional[float] = None,
    ) -> float:
        """
        Обновление после получения РЕАЛЬНОГО наблюдения.

        Parameters
        ----------
        model_logprob : float
            log P_model(w_t | context) — логарифм вероятности
            наблюдения по текущей модели.
        baseline_logprob : float | None
            log P_baseline(w_t). Если None — используется
            равномерное или настроенное при инициализации.

        Returns
        -------
        float : предиктивный выигрыш за этот шаг.
        """
        blp = baseline_logprob if baseline_logprob is not None else self._log_baseline
        step_pg = model_logprob - blp
        self.npg += step_pg
        self.n_steps += 1
        self.step_gains.append(step_pg)
        if len(self.step_gains) > 1000:
            self.step_gains.pop(0)
        return step_pg

    @property
    def avg_pg(self) -> float:
        return self.npg / max(self.n_steps, 1)

    @property
    def scale(self) -> float:
        """UAF-шкала [-3, +3]."""
        return math.tanh(self.avg_pg / 2.0) * 3.0

    @property
    def scale_label(self) -> str:
        return self.SCALE_LABELS.get(round(self.scale), f"{self.scale:.2f}")

    def recent_trend(self, window: int = 20) -> str:
        """Тренд: 'improving', 'stable', 'degrading'."""
        if len(self.step_gains) < window * 2:
            return "insufficient_data"
        recent = sum(self.step_gains[-window:]) / window
        earlier = sum(self.step_gains[-window*2:-window]) / window
        if recent > earlier + 0.05:
            return "improving"
        elif recent < earlier - 0.05:
            return "degrading"
        return "stable"

    def summary(self) -> Dict[str, object]:
        return {
            "npg": round(self.npg, 4),
            "avg_pg": round(self.avg_pg, 4),
            "scale": round(self.scale, 3),
            "scale_label": self.scale_label,
            "n_steps": self.n_steps,
            "trend": self.recent_trend(),
        }


# ============================================================
# LYAPUNOV STABILITY MONITOR
# ============================================================

class LyapunovMonitor:
    """
    Следит за стабильностью познания через функцию Ляпунова.

    V(P_t) = D_KL(P_t ‖ P_true)

    Теорема 11.1 (THEORY.md §11):
        При условиях Роббинса-Монро (Σα_t = ∞, Σα_t² < ∞)
        V(P_t) → 0, то есть P_t → P_true почти наверное.

    Практически: мы не знаем P_true, поэтому используем
    валидационную потерю как прокси.
    """

    def __init__(self):
        self.v_history: List[float] = []
        self.decreasing_count: int = 0
        self.increasing_count: int = 0

    def update(self, current_val_loss: float) -> Dict[str, object]:
        """
        Обновление монитора по валидационной потере.
        Предполагаем: val_loss ≈ D_KL(P_t ‖ P_true) + const.
        """
        self.v_history.append(current_val_loss)

        stable = True
        rho = None
        if len(self.v_history) >= 2:
            prev, curr = self.v_history[-2], self.v_history[-1]
            if prev > 1e-6:
                rho = curr / prev
                if rho < 1.0:
                    self.decreasing_count += 1
                    self.increasing_count = max(0, self.increasing_count - 1)
                else:
                    self.increasing_count += 1
                    self.decreasing_count = max(0, self.decreasing_count - 1)
                stable = rho <= 1.0

        return {
            "v": current_val_loss,
            "rho": round(rho, 4) if rho is not None else None,
            "stable": stable,
            "converging": self.decreasing_count > self.increasing_count,
            "diverging": self.increasing_count >= 5,
        }

    def knowledge_horizon(self, target_k: float) -> Optional[int]:
        """
        τ_K = min{t : V(t) < K} — горизонт знания (§12 THEORY.md).
        Возвращает шаг, когда была достигнута точность K, или None.
        """
        for t, v in enumerate(self.v_history):
            if v < target_k:
                return t
        return None


# ============================================================
# БЫСТРЫЕ ПРОВЕРКИ
# ============================================================

def _run_checks():
    print("free_energy.py checks...")

    # 1. F = U − T·H
    assert abs(compute_free_energy(2.0, 1.0, 0.8) - 1.2) < 1e-6
    assert compute_free_energy(0.5, 1.5, 0.8) < 0  # T·H > U → F < 0
    print("  ✓ F = U − T·H")

    # 2. Энтропия максимальна при равномерном распределении
    uniform = {"a": 0.25, "b": 0.25, "c": 0.25, "d": 0.25}
    peaked = {"a": 0.97, "b": 0.01, "c": 0.01, "d": 0.01}
    assert entropy(uniform) > entropy(peaked)
    assert abs(normalized_entropy(uniform) - 1.0) < 0.01
    print("  ✓ Entropy is maximal for uniform distribution")

    # 3. KL-дивергенция неотрицательна и несимметрична
    p = {"a": 0.9, "b": 0.05, "c": 0.05}
    q = {"a": 0.1, "b": 0.8,  "c": 0.1}
    assert kl_divergence(p, q) >= 0
    assert kl_divergence(q, p) >= 0
    assert abs(kl_divergence(p, q) - kl_divergence(q, p)) > 0.1   # несимметрична
    assert kl_divergence(p, p) < 1e-9  # D_KL(P‖P) = 0
    print("  ✓ KL divergence: non-negative, asymmetric, D_KL(P‖P)=0")

    # 4. Tracker обновляется корректно
    tracker = FreeEnergyTracker(residuum_alpha=0.5)
    dist = {"a": 0.8, "b": 0.2}
    state = tracker.update(2.0, dist)
    assert state.step == 1
    assert state.U == 2.0
    assert 0.0 < state.H < 1.0
    assert state.F == compute_free_energy(state.U, state.T, state.H)
    print("  ✓ FreeEnergyTracker updates correctly")

    # 5. NPG после наблюдения
    pg = PredictiveGainTracker(vocab_size=100)
    model_lp = math.log(0.5)
    base_lp = math.log(0.01)
    gain = pg.update(model_lp, base_lp)
    assert gain > 0  # модель лучше базовой
    assert pg.scale > 0
    print(f"  ✓ PG is positive when model beats baseline: {gain:.3f}")

    # 6. NPG < 0 когда модель хуже базовой
    pg2 = PredictiveGainTracker(vocab_size=100)
    gain2 = pg2.update(math.log(0.001), math.log(0.01))
    assert gain2 < 0
    print(f"  ✓ PG is negative when model underperforms: {gain2:.3f}")

    # 7. Ляпунов мониторинг
    lm = LyapunovMonitor()
    for v in [1.0, 0.8, 0.6, 0.4, 0.2]:
        r = lm.update(v)
    assert r["converging"]
    assert lm.knowledge_horizon(0.5) == 3  # достигли < 0.5 на шаге 3
    print("  ✓ LyapunovMonitor detects convergence")

    print("\nAll free_energy.py checks passed. ✓")


if __name__ == "__main__":
    _run_checks()
