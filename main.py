import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.fft import fft, ifft
import os

def read_wav_and_validate(filename):
    try:
        sample_rate, data = wavfile.read(filename)
        if len(data.shape) > 1:
            data = data[:, 0]
        return sample_rate, data.astype(np.float64)

    except FileNotFoundError:
        print(f"файл не найден")
        return None, None

    except Exception as e:
        print(f"ошибка при чтения")
        return None, None

def get_sample_count(total_samples):
    while True:
        try:
            count = int(input(f"Введите количество отсчетов (1-{total_samples}): "))
            if 1 <= count <= total_samples:
                return count
            else:
                print(f"Число должно быть от 1 до {total_samples}")
        except ValueError:
            print("Ошибка: введите целое число!")

def plot_scatter(time, samples, count, sample_rate):
    plt.figure(figsize=(12, 6))
    plt.scatter(time[:count], samples[:count], marker='*', s=20, c='blue', alpha=0.7)
    plt.xlabel('Время (секунды)', fontsize=12)
    plt.ylabel('Амплитуда', fontsize=12)
    plt.title(f'Точечный график звукового сигнала (первые {count} отсчетов)', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_oscillogram(time, samples, sample_rate):
    plt.figure(figsize=(14, 6))
    plt.plot(time, samples, linewidth=0.8, color='green')
    plt.xlabel('Время (секунды)', fontsize=12)
    plt.ylabel('Амплитуда', fontsize=12)
    plt.title('Осциллограмма звукового сигнала', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def compute_cepstrum(samples, sample_rate):
    spectrum = fft(samples)
    power_spectrum = np.abs(spectrum) ** 2
    power_spectrum = np.maximum(power_spectrum, 1e-10)
    log_power = np.log(power_spectrum)
    cepstrum = np.real(ifft(log_power))
    quefrency = np.arange(len(cepstrum)) / sample_rate
    return cepstrum[:len(cepstrum) // 2], quefrency[:len(cepstrum) // 2]

def plot_cepstrum(cepstrum, quefrency, sample_rate):
    plt.figure(figsize=(14, 6))
    #ограничиваю диапазон для наглядности
    max_time = min(3.0, quefrency[-1])
    mask = quefrency <= max_time
    plt.plot(quefrency[mask], cepstrum[mask], linewidth=1.5, color='red')
    plt.xlabel('Квефренция (секунды)', fontsize=12)
    plt.ylabel('Амплитуда кепстра', fontsize=12)
    plt.title('Кепстральный анализ сигнала\nОДПФ[ln(|F(ω)|²)]', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_histogram(samples, sample_rate):
    #определяю количество интервалов по правилу Стёрджеса
    num_bins = int(1 + np.log2(len(samples)))
    plt.figure(figsize=(12, 6))
    counts, bins, patches = plt.hist(samples, bins=num_bins, alpha=0.7, color='purple', edgecolor='black')
    plt.xlabel('Амплитуда', fontsize=12)
    plt.ylabel('Частота попаданий', fontsize=12)
    plt.title('Гистограмма распределения амплитуд отсчетов', fontsize=14)
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.show()


def main():
    while True:
        filename = input("\nВведите имя файла: ").strip()
        if filename:
            break

    sample_rate, samples = read_wav_and_validate(filename)

    if sample_rate is None or samples is None:
        print("Программа завершена из-за ошибки.")
        return

    count = get_sample_count(len(samples))
    time = np.arange(len(samples)) / sample_rate
    plot_scatter(time, samples, count, sample_rate)

    plot_oscillogram(time, samples, sample_rate)

    cepstrum, quefrency = compute_cepstrum(samples, sample_rate)
    plot_cepstrum(cepstrum, quefrency, sample_rate)

    plot_histogram(samples, sample_rate)

if __name__ == "__main__":
    main()
