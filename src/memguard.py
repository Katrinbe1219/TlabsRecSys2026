"""Защита от нехватки памяти: WSL с ~7.6 ГБ может упасть целиком, если процесс съест всё

Перед каждым тяжёлым шагом вызываем check(): если свободной памяти меньше порога,
шаг не начинается, а скрипт падает с понятной ошибкой 
"""
import resource


def available_gb() -> float:
    """MemAvailable из /proc/meminfo: сколько памяти система может отдать без свопа"""
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) / 1024**2
    return float("nan")


def peak_rss_gb() -> float:
    """Пиковая память текущего процесса с момента старта"""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2


def check(min_free_gb: float = 2.5, what: str = "") -> None:
    free = available_gb()
    if free < min_free_gb:
        raise MemoryError(f"Свободно {free:.1f} ГБ < {min_free_gb} ГБ — останавливаюсь перед шагом «{what}», "
                          f"чтобы не уронить WSL. Закройте другие программы и перезапустите")
