"""
UAF LLM Agent — языковая модель с предиктивным кодированием
основанная на принципе минимизации свободной энергии.

Формальные основания (THEORY.md):

    F  = U − T·H                          (свободная энергия Фристона)
    U  = 𝔼[−log P(w_{t+1} | w_{≤t})]     (энергия = среднее удивление)
    H  = −Σ P(x) log P(x)                 (энтропия логитов = мера сомнения)
    T  = f(residuum)                       (температура = накопленная ошибка)

    NPG(M, T) = Σ log M(wₜ|w_{<t}) / M₀(wₜ|w_{<t})   (предиктивный выигрыш)
    doubt     = g(residuum, H)             (активное сомнение)

    Aufhebung при: doubt > θ               (диалектический синтез: §5 THEORY.md)

    UAF-шкала: NPG нормируется в [-3, +3]  (§3 THEORY.md)

Ключевое отличие от стандартного обучения:
    Модель не переобучается → она начинает сомневаться → синтезирует новое состояние.
    Это не регуляризация. Это предиктивная эпистемология.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass, field

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False


# ============================================================
# DATACLASS: UAF-состояние
# ============================================================

@dataclass
class UAFState:
    """
    Полное состояние UAF-системы в момент времени t.

    Все поля соответствуют формулам из THEORY.md.
    """
    step: int = 0

    # Компоненты свободной энергии
    free_energy: float = 1.0    # F = U − T·H
    energy: float = 1.0         # U = 𝔼[−log P] = среднее удивление
    entropy: float = 1.0        # H = −Σ P log P (энтропия распределения логитов)
    temperature: float = 1.0    # T ∈ (0, ∞), растёт с накопленной ошибкой

    # Производные метрики
    residuum: float = 0.2       # накопленное удивление (EMA)
    doubt: float = 0.3          # активное сомнение = g(residuum, H)
    surprisal: float = 1.0      # −log P текущего шага

    # Предиктивный выигрыш
    npg: float = 0.0            # накопленный NPG относительно базовой модели
    uaf_scale: float = 0.0      # нормированная шкала [-3, +3]

    # Диалектика
    aufhebung_count: int = 0
    last_aufhebung_step: int = -999

    # История (ограничена max_history)
    history: Dict[str, List[float]] = field(default_factory=lambda: {
        "free_energy": [], "energy": [], "entropy": [], "temperature": [],
        "residuum": [], "doubt": [], "loss": [], "npg": [], "effective_lr": []
    })

    def to_dict(self) -> dict:
        """Сериализация для checkpoint."""
        return {k: v for k, v in self.__dict__.items() if k != "history"}

    @classmethod
    def from_dict(cls, d: dict) -> "UAFState":
        s = cls()
        for k, v in d.items():
            if hasattr(s, k):
                setattr(s, k, v)
        return s


# ============================================================
# ОСНОВНОЙ КЛАСС
# ============================================================

class UAFLLM(nn.Module):
    """
    UAF LLM — обёртка над CausalLM с UAF-динамикой.

    Параметры
    ---------
    base_model_name : str
        Имя HuggingFace модели (default: "gpt2")
    lr : float
        Базовый learning rate. Эффективный LR снижается при росте doubt:
            lr_eff = lr / (1 + doubt · α)
    doubt_threshold : float
        Порог сомнения для запуска Aufhebung. Рекомендуется 1.2–1.6.
    aufhebung_cooldown : int
        Минимальный интервал между Aufhebung-синтезами (шагов).
    aufhebung_noise_scale : float
        Масштаб шума при синтезе. Малый → консервативный синтез.
    residuum_ema : float
        Коэффициент EMA для накопления резидуума (0 < α < 1).
    max_history : int
        Максимальная длина истории метрик в памяти.
    baseline_uniform : bool
        Если True, базовая модель для NPG — равномерное распределение по словарю.
    """

    def __init__(
        self,
        base_model_name: str = "gpt2",
        device: Optional[str] = None,
        lr: float = 5e-5,
        doubt_threshold: float = 1.4,
        aufhebung_cooldown: int = 200,
        aufhebung_noise_scale: float = 0.008,
        residuum_ema: float = 0.97,
        max_history: int = 500,
        baseline_uniform: bool = True,
    ):
        super().__init__()

        # Устройство
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        # Гиперпараметры
        self.base_lr = lr
        self.doubt_threshold = doubt_threshold
        self.aufhebung_cooldown = aufhebung_cooldown
        self.aufhebung_noise_scale = aufhebung_noise_scale
        self.residuum_ema = residuum_ema
        self.max_history = max_history
        self.lr_damping = 1.5  # коэффициент подавления LR при росте doubt

        # Загрузка модели
        if not HAS_TRANSFORMERS:
            raise ImportError("transformers not installed: pip install transformers")

        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(base_model_name).to(device)

        # Размер словаря для базовой NPG-модели
        self.vocab_size = self.model.config.vocab_size
        self._log_baseline = -math.log(self.vocab_size) if baseline_uniform else None

        # UAF-состояние
        self.uaf = UAFState()
        self.uaf.last_aufhebung_step = -aufhebung_cooldown

        # Оптимизатор
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.base_lr)

    # ----------------------------------------------------------
    # ВЫЧИСЛЕНИЕ ЭНТРОПИИ ЛОГИТОВ
    # ----------------------------------------------------------

    def _entropy_of_logits(self, logits: torch.Tensor) -> float:
        """
        H = −Σ P(x) log P(x)  по последнему слою логитов.

        Это мера 'сомнения' модели относительно следующего токена:
        - H ≈ 0: модель уверена (знает следующий токен)
        - H ≈ log(V): модель полностью неуверена (равномерное распределение)
        """
        # Берём последний токен последнего элемента батча
        last_logits = logits[0, -1, :]  # (vocab_size,)
        probs = F.softmax(last_logits, dim=-1)
        log_probs = F.log_softmax(last_logits, dim=-1)
        H = -(probs * log_probs).sum().item()
        # Нормируем в [0, 1] относительно максимальной энтропии
        H_max = math.log(self.vocab_size)
        return H / H_max

    # ----------------------------------------------------------
    # ВЫЧИСЛЕНИЕ NPG (предиктивный выигрыш)
    # ----------------------------------------------------------

    def _compute_step_npg(self, log_probs_model: torch.Tensor,
                           labels: torch.Tensor) -> float:
        """
        NPG_step = Σ log[M(wₜ|w_{<t}) / M₀(wₜ|w_{<t})]

        Базовая модель M₀ — равномерное распределение (log M₀ = -log V).

        NPG > 0: модель предсказывает лучше случайного.
        NPG = 0: модель не лучше случайного.
        NPG < 0: модель предсказывает хуже случайного (дистрибуционный сдвиг).
        """
        if self._log_baseline is None:
            return 0.0

        shift_log_probs = log_probs_model[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        # log P_model(wₜ)
        gathered = shift_log_probs.view(-1, self.vocab_size).gather(
            1, shift_labels.view(-1, 1)
        ).squeeze(-1)  # (T,)

        # log PG_step = Σ [log_model - log_baseline]
        step_npg = (gathered - self._log_baseline).mean().item()
        return step_npg

    # ----------------------------------------------------------
    # ОБНОВЛЕНИЕ UAF-СОСТОЯНИЯ
    # ----------------------------------------------------------

    def _update_uaf_state(self, surprisal: float, entropy_H: float, step_npg: float):
        """
        Обновление всех компонент UAF согласно THEORY.md §2–3.

            U = surprisal (текущее удивление, скаляр)
            H = entropy_H (нормированная энтропия логитов)
            residuum(t) = α·residuum(t-1) + (1-α)·U   (EMA)
            T = 0.5 + residuum                           (температура)
            F = U − T·H                                  (свободная энергия)
            doubt = β·H + γ·residuum                    (активное сомнение)

        Ключевой инвариант: doubt растёт при высокой неопределённости И
        высоком накопленном удивлении. Это соответствует §4 THEORY.md:
        антитезис = случаи где P(y|x) < θ.
        """
        s = self.uaf
        α = self.residuum_ema

        # Накопление резидуума (EMA удивления)
        s.surprisal = surprisal
        s.residuum = α * s.residuum + (1 - α) * surprisal

        # Температура T: растёт с накопленной ошибкой, минимум 0.3
        s.temperature = max(0.3, 0.5 + s.residuum * 0.8)

        # Энтропия H (уже нормирована в [0,1])
        s.entropy = entropy_H

        # Энергия U — нормализованное удивление
        s.energy = min(surprisal, 10.0)

        # Свободная энергия F = U − T·H
        s.free_energy = s.energy - s.temperature * s.entropy

        # Сомнение = взвешенная сумма H и residuum
        # Это активный порог, управляющий Aufhebung
        s.doubt = 0.4 * s.entropy + 0.6 * min(s.residuum, 2.0)

        # Накопленный NPG и UAF-шкала [-3, +3]
        s.npg += step_npg
        # Нормируем: NPG ≈ log(V)/step ~ 0–8 per step, грубо масштабируем в [-3,+3]
        norm_npg = math.tanh(s.npg / max(s.step, 1) / 2.0) * 3
        s.uaf_scale = round(norm_npg, 2)

        # Запись истории
        h = s.history
        h["free_energy"].append(s.free_energy)
        h["energy"].append(s.energy)
        h["entropy"].append(s.entropy)
        h["temperature"].append(s.temperature)
        h["residuum"].append(s.residuum)
        h["doubt"].append(s.doubt)
        h["npg"].append(s.npg)

        # Ограничиваем длину истории
        for key in h:
            if len(h[key]) > self.max_history:
                h[key] = h[key][-self.max_history:]

    # ----------------------------------------------------------
    # AUFHEBUNG — диалектический синтез
    # ----------------------------------------------------------

    def aufhebung(self, force: bool = False) -> bool:
        """
        Aufhebung: L_{k+1} = L_k ∪ {новые переменные, объясняющие резидуум L_k}

        Реализация §5 THEORY.md:
        - Тезис  M_k        = текущие веса
        - Антитезис R(t)    = накопленный резидуум (high doubt)
        - Синтез M_{k+1}   = M_k + perturbation · T

        Perturbation масштабируется температурой: чем выше неопределённость,
        тем смелее синтез. Это нелинейный выход из локального минимума.

        Защиты:
        - cooldown: не чаще чем раз в aufhebung_cooldown шагов
        - force: принудительный вызов (например, из внешнего цикла)
        """
        s = self.uaf

        if not force and s.doubt < self.doubt_threshold:
            return False
        if s.step - s.last_aufhebung_step < self.aufhebung_cooldown:
            return False

        s.aufhebung_count += 1
        s.last_aufhebung_step = s.step

        prev_doubt = s.doubt
        prev_residuum = s.residuum

        # Синтез: возмущение весов масштабировано температурой
        noise_scale = self.aufhebung_noise_scale * s.temperature
        with torch.no_grad():
            for param in self.model.parameters():
                # Направленный шум: знак определяется градиентом (если есть)
                noise = torch.randn_like(param) * noise_scale
                param.add_(noise)

        # Сброс метасостояния после синтеза (§5.1 THEORY.md)
        s.doubt = 0.35
        s.residuum = max(0.15, s.residuum * 0.4)
        s.entropy = 0.7
        s.temperature = 0.6
        s.free_energy = s.energy - s.temperature * s.entropy

        print(
            f"[Aufhebung #{s.aufhebung_count}] "
            f"step={s.step} | "
            f"doubt: {prev_doubt:.3f}→{s.doubt:.3f} | "
            f"residuum: {prev_residuum:.3f}→{s.residuum:.3f} | "
            f"noise_scale={noise_scale:.5f}"
        )
        return True

    # ----------------------------------------------------------
    # ШАГ ОБУЧЕНИЯ
    # ----------------------------------------------------------

    def train_step(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, float]:
        """
        Один шаг обучения с полной UAF-динамикой.

        Порядок операций:
        1. Forward pass → logits, loss (стандартный CausalLM CE)
        2. Вычислить H (энтропию логитов) → measure of doubt
        3. Вычислить NPG_step относительно базовой модели
        4. Обновить UAF-состояние (F, T, H, residuum, doubt)
        5. Скорректировать LR: lr_eff = lr / (1 + doubt · α)
        6. Backward pass
        7. Проверить порог Aufhebung → синтез если нужен

        Возвращает
        ----------
        dict с полными метриками UAF и training.
        """
        self.train()
        self.uaf.step += 1

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=input_ids,
        )

        logits = outputs.logits   # (B, T, V)
        loss = outputs.loss       # scalar (CE)

        # --- UAF-метрики ---
        surprisal = loss.item()
        entropy_H = self._entropy_of_logits(logits.detach())

        # Log-probs для NPG
        log_probs = F.log_softmax(logits.detach(), dim=-1)
        step_npg = self._compute_step_npg(log_probs, input_ids)

        # Обновление состояния
        self._update_uaf_state(surprisal, entropy_H, step_npg)

        # --- Динамический LR ---
        # Чем выше doubt, тем осторожнее обновляем веса
        effective_lr = self.base_lr / (1.0 + self.uaf.doubt * self.lr_damping)
        for pg in self.optimizer.param_groups:
            pg["lr"] = effective_lr

        # --- Backward ---
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()

        # --- Aufhebung ---
        aufhebung_fired = self.aufhebung()

        # Сохранение в историю
        self.uaf.history["loss"].append(loss.item())
        self.uaf.history["effective_lr"].append(effective_lr)

        return {
            # Training
            "loss":             loss.item(),
            "effective_lr":     effective_lr,
            # Компоненты F = U − T·H
            "free_energy":      self.uaf.free_energy,
            "energy_U":         self.uaf.energy,
            "entropy_H":        self.uaf.entropy,
            "temperature_T":    self.uaf.temperature,
            # Сомнение и накопленная ошибка
            "residuum":         self.uaf.residuum,
            "doubt":            self.uaf.doubt,
            # Предиктивный выигрыш
            "step_npg":         step_npg,
            "npg_cumulative":   self.uaf.npg,
            "uaf_scale":        self.uaf.uaf_scale,
            # Диалектика
            "aufhebung_count":  self.uaf.aufhebung_count,
            "aufhebung_fired":  aufhebung_fired,
        }

    # ----------------------------------------------------------
    # ГЕНЕРАЦИЯ
    # ----------------------------------------------------------

    @torch.no_grad()
    def generate(self, prompt: str, max_new_tokens: int = 60, **kwargs) -> str:
        """
        Генерация с UAF-температурой.

        Когда doubt высок (модель сомневается), температура softmax растёт:
        разнообразие ответов увеличивается — это аналог «сомнения» при ответе.
        Когда doubt низок (уверенная модель), температура снижается.
        """
        self.eval()
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        # UAF-температура генерации: [0.5, 2.0]
        gen_temperature = max(0.5, min(2.0, 0.6 + self.uaf.doubt * 0.9))

        output_ids = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=gen_temperature,
            do_sample=True,
            top_p=0.92,
            repetition_penalty=1.1,
            **kwargs,
        )
        return self.tokenizer.decode(output_ids[0], skip_special_tokens=True)

    # ----------------------------------------------------------
    # ДИАГНОСТИКА
    # ----------------------------------------------------------

    def report(self) -> str:
        """Человекочитаемый отчёт о текущем UAF-состоянии."""
        s = self.uaf
        uaf_scale_str = {
            3: "+3 (уникально предсказывает)", 2: "+2 (лучше всех доступных)",
            1: "+1 (лучше базовой)", 0: " 0 (как случайность)",
            -1: "-1 (хуже базовой)", -2: "-2 (анти-предсказание)",
            -3: "-3 (оптимизирован на ошибки)"
        }.get(round(s.uaf_scale), f"{s.uaf_scale:.2f}")

        lines = [
            "=" * 60,
            f"UAF STATE  |  Step {s.step}",
            "=" * 60,
            f"  F  = U − T·H  =  {s.energy:.4f} − {s.temperature:.3f}·{s.entropy:.4f}  =  {s.free_energy:.4f}",
            f"  Residuum:  {s.residuum:.4f}",
            f"  Doubt:     {s.doubt:.4f}  (threshold: {self.doubt_threshold})",
            f"  NPG:       {s.npg:.3f}  → UAF scale: {uaf_scale_str}",
            f"  Aufhebung: {s.aufhebung_count} синтезов",
            "=" * 60,
        ]
        return "\n".join(lines)

    # ----------------------------------------------------------
    # CHECKPOINT
    # ----------------------------------------------------------

    def save_checkpoint(self, path: str):
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "uaf_state": self.uaf.to_dict(),
            "history": self.uaf.history,
            "config": {
                "lr": self.base_lr,
                "doubt_threshold": self.doubt_threshold,
                "aufhebung_cooldown": self.aufhebung_cooldown,
                "aufhebung_noise_scale": self.aufhebung_noise_scale,
                "residuum_ema": self.residuum_ema,
            }
        }, path)
        print(f"Checkpoint saved → {path}")

    def load_checkpoint(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.uaf = UAFState.from_dict(ckpt["uaf_state"])
        self.uaf.history = ckpt.get("history", self.uaf.history)
        print(f"Checkpoint loaded ← {path}")
        print(self.report())

    def reset_uaf_state(self):
        """Сброс UAF-состояния без сброса весов модели."""
        cooldown = self.aufhebung_cooldown
        self.uaf = UAFState()
        self.uaf.last_aufhebung_step = -cooldown


# ============================================================
# УТИЛИТА: обучение без HuggingFace (для тестов и демо)
# ============================================================

class UAFToyModel:
    """
    Лёгкая симуляция UAF-динамики без реальной LLM.
    Используется в тестах и HTML-демо.

    Модель: y = w·x + b, задача: предсказать линейную зависимость.
    UAF-состояние рассчитывается аналитически по тем же формулам.
    """

    def __init__(self, doubt_threshold: float = 1.2, vocab_size: int = 256):
        self.doubt_threshold = doubt_threshold
        self.vocab_size = vocab_size
        self._log_baseline = -math.log(vocab_size)

        # Веса
        self.w = 1.0 + (0.5 - __import__('random').random()) * 0.4
        self.b = 0.0

        # UAF-состояние
        self.uaf = UAFState()
        self.uaf.last_aufhebung_step = -500

        # История
        self.train_losses: List[float] = []
        self.val_losses: List[float] = []
        self.aufhebung_steps: List[int] = []

        # LR
        self.base_lr = 0.02

    def _sample_batch(self, noise: float = 0.05):
        import random
        x = random.uniform(-2, 2)
        y = 2.0 * x + random.gauss(0, noise)
        return x, y

    def step(self, inject_anomaly: bool = False) -> Dict[str, float]:
        import random
        self.uaf.step += 1

        x, y = self._sample_batch(noise=0.05 if not inject_anomaly else 0.9)
        pred = self.w * x + self.b
        surprisal = (pred - y) ** 2

        # Энтропия симулируется через неопределённость предсказания
        # H ≈ tanh(|ошибка|) — больше ошибка, больше неопределённость
        entropy_H = math.tanh(abs(pred - y) * 0.8)

        # NPG: toy baseline — константное предсказание 0
        baseline_loss = y ** 2
        step_npg = math.log(max(baseline_loss, 1e-8)) - math.log(max(surprisal, 1e-8))

        # Обновление UAF
        α = 0.95
        s = self.uaf
        s.surprisal = surprisal
        s.residuum = α * s.residuum + (1 - α) * surprisal
        s.temperature = max(0.3, 0.5 + s.residuum * 0.8)
        s.entropy = entropy_H
        s.energy = min(surprisal, 10.0)
        s.free_energy = s.energy - s.temperature * s.entropy
        s.doubt = 0.4 * s.entropy + 0.6 * min(s.residuum, 2.0)
        s.npg += step_npg
        s.uaf_scale = round(math.tanh(s.npg / max(s.step, 1) / 2.0) * 3, 2)

        # Динамический LR
        effective_lr = self.base_lr / (1.0 + s.doubt * 1.5)

        # Обновление весов
        self.w -= effective_lr * 2 * (pred - y) * x
        self.b -= effective_lr * 2 * (pred - y)
        self.w = max(-5, min(5, self.w))

        # Aufhebung
        aufhebung_fired = False
        if s.doubt > self.doubt_threshold and s.step - s.last_aufhebung_step > 200:
            s.aufhebung_count += 1
            s.last_aufhebung_step = s.step
            self.aufhebung_steps.append(s.step)
            self.w += random.gauss(0, 0.08 * s.temperature)
            self.b += random.gauss(0, 0.04 * s.temperature)
            s.doubt = 0.35
            s.residuum *= 0.4
            s.entropy = 0.7
            s.temperature = 0.6
            aufhebung_fired = True

        # Val loss
        x_val, y_val = self._sample_batch(noise=0.02)
        val_pred = self.w * x_val + self.b
        val_loss = (val_pred - y_val) ** 2

        self.train_losses.append(surprisal)
        self.val_losses.append(val_loss)

        return {
            "step": s.step,
            "train_loss": surprisal,
            "val_loss": val_loss,
            "free_energy": s.free_energy,
            "energy_U": s.energy,
            "entropy_H": s.entropy,
            "temperature_T": s.temperature,
            "residuum": s.residuum,
            "doubt": s.doubt,
            "npg": s.npg,
            "uaf_scale": s.uaf_scale,
            "effective_lr": effective_lr,
            "aufhebung_count": s.aufhebung_count,
            "aufhebung_fired": aufhebung_fired,
        }

    def reset(self):
        import random
        self.w = 1.0 + (0.5 - random.random()) * 0.4
        self.b = 0.0
        self.uaf = UAFState()
        self.uaf.last_aufhebung_step = -500
        self.train_losses = []
        self.val_losses = []
        self.aufhebung_steps = []


# ============================================================
# ПРИМЕР ИСПОЛЬЗОВАНИЯ
# ============================================================

if __name__ == "__main__":
    import sys

    # ── Режим 1: игрушечная модель (всегда доступна) ──────────────
    print("=== UAF Toy Model Demo ===")
    toy = UAFToyModel(doubt_threshold=1.2)

    for step in range(300):
        inject = (step == 150)  # аномалия на шаге 150
        m = toy.step(inject_anomaly=inject)

        if step % 50 == 0 or m["aufhebung_fired"]:
            tag = " ← ANOMALY" if inject else ""
            auf = f" [Aufhebung #{m['aufhebung_count']}]" if m["aufhebung_fired"] else ""
            print(
                f"Step {m['step']:4d} | "
                f"loss={m['train_loss']:.4f} | "
                f"F={m['free_energy']:.4f} | "
                f"doubt={m['doubt']:.3f} | "
                f"UAF={m['uaf_scale']:+.1f} | "
                f"Aufhebung×{m['aufhebung_count']}"
                f"{auf}{tag}"
            )

    print(f"\nFinal NPG: {toy.uaf.npg:.3f}")
    print(f"Final UAF scale: {toy.uaf.uaf_scale:+.2f}")
    print(f"Aufhebung total: {toy.uaf.aufhebung_count}")

    # ── Режим 2: реальная LLM (если доступен transformers) ───────
    if HAS_TRANSFORMERS and "--llm" in sys.argv:
        print("\n=== UAF LLM Demo (GPT-2) ===")
        model = UAFLLM(
            base_model_name="gpt2",
            lr=5e-5,
            doubt_threshold=1.4,
            aufhebung_cooldown=100,
        )

        texts = [
            "Машинное обучение — наука о предсказании паттернов в данных.",
            "Свободная энергия минимизируется во всех адаптивных системах.",
            "UAF — это не регуляризация, это предиктивная эпистемология.",
            "Диалектика Гегеля и байесовское обновление — одна и та же операция.",
        ]

        encodings = model.tokenizer(
            texts, return_tensors="pt", padding=True,
            truncation=True, max_length=64
        ).to(model.device)

        for epoch in range(3):
            ids = encodings["input_ids"]
            mask = encodings["attention_mask"]
            m = model.train_step(ids, mask)
            print(
                f"Epoch {epoch+1}: "
                f"loss={m['loss']:.4f} | "
                f"F={m['free_energy']:.4f} | "
                f"doubt={m['doubt']:.3f} | "
                f"NPG={m['step_npg']:.3f} | "
                f"scale={m['uaf_scale']:+.2f}"
            )

        print(model.report())
        print("\nGeneration:", model.generate("Свободная энергия —"))
