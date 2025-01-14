"""Random data generator"""
import random
import csv
import numpy as np
import matplotlib.pyplot as plt
from sympy import nextprime
from tqdm import tqdm
from scipy.special import gammainc
from functools import wraps
from time import time


def prime_diriv(n):
    ans = set()
    d = 2
    while d * d <= n:
        if n % d == 0:
            ans.add(d)
            n //= d
        else:
            d += 1
    if n > 1:
        ans.add(n)
    return ans


def find_generator_of_group_z(p):
    all_prime_dir_p = prime_diriv(p-1)
    g = 2
    while True:
        is_generator = True
        for q in all_prime_dir_p:
            if pow(g, p // q, p + 1) == 1:
                is_generator = False
                g += 1
                break
            if is_generator:
                return g


def mean_std_cvar_uniform_randomness(list_of_sets):
    with open("statisitc_for_each_set.csv", 'w',
              encoding='utf-8') as csvfile:
        # creating a csv writer object
        BLOCK_SIZE = 128
        ALPHA = 0.01
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(["Среднее", "Среднеквадратичное отклонение",
                            "Коэффициент вариации",
                            "Принята гипотеза о равномерном распределении",
                            f"Пройден ли частотный тест для блоков размера"
                            f" {BLOCK_SIZE}, p_value >= {ALPHA}"""])
        for bin_set in list_of_sets:
            float_set = [float(el) for el in bin_set]
            mean_val = np.mean(float_set)
            std_val = np.std(float_set)
            std_cvar = mean_val / std_val

            number_of_zeros = np.sum(float_set)
            number_of_ones = len(float_set) - number_of_zeros

            number_of_ones_or_zeros_theor = 1 / 2 * len(float_set)

            xi_2 = ((number_of_zeros - number_of_ones_or_zeros_theor)**2
                    / number_of_ones_or_zeros_theor +
                    (number_of_ones - number_of_ones_or_zeros_theor)**2
                    / number_of_ones_or_zeros_theor)

            # xi^2_ при alpha = 0.01 и k = 1: 6.635

            if xi_2 < 6.635:
                ravnomerno = f"Да, xi^2 = {round(xi_2, 3)}"
            else:
                ravnomerno = f"Нет, xi^2 = {round(xi_2, 3)}"

            freq_xi_2,  freq_p_value = frequency_test_within_block(float_set,
                                                                   BLOCK_SIZE)
            freq_res = (f", xi^2 = {round(freq_xi_2, 3)},"
                        f" p_value = {round(freq_p_value, 3)}")

            if freq_p_value >= ALPHA:
                freq_block_random = "Да"
            else:
                freq_block_random = "Нет"
            freq_block_random += freq_res

            csvwriter.writerow([round(mean_val, 3), round(std_val, 3),
                                round(std_cvar, 3),
                                ravnomerno, freq_block_random])


def frequency_test_within_block(bin_set, m):
    set_fragments = [bin_set[el:el+m] for el in
                     range(len(bin_set)//m)]
    p_freq = [np.mean(el) for el in set_fragments]
    xi_2 = 4 * m * np.sum([pow(el - 1/2, 2) for el in p_freq])
    # xi^2_ при alpha = 0.01 и k = 1: 6.635
    p_value = 1 - gammainc(len(bin_set)//m / 2, xi_2/2)
    xi_2 = round(xi_2, 3)

    return xi_2, p_value


def timing(f):
    @wraps(f)
    def wrap(*args, **kw):
        ts = time()
        _ = f(*args, **kw)
        te = time()
        time_spend = te-ts
        # print(f'{te-ts:.4f} sec')
        return time_spend
    return wrap


def generate_random_list_of_sets(number_of_elements, p, g):
    bit_output = ""
    seed = random.randint(1, 1e14)
    x = seed
    for _ in range(number_of_elements):
        x = pow(g, x, p)
        b = 0 if x < p/2 else 1
        bit_output += str(b)
    return bit_output


@timing
def generate_random_list_of_sets_timed(number_of_elements, p, g):
    return generate_random_list_of_sets(number_of_elements, p, g)


@timing
def standart_genration(number_of_elements):
    random.seed(random.randint(1, 1e14))
    generated_str = ""
    for _ in range(number_of_elements):
        generated_str += str(random.randint(0, 1))
    return generated_str


if __name__ == "__main__":
    NUMBER_OF_ELEMENTS = 10000
    NUMBER_OF_SETS = 40
    NUMBER_BEFORE = 3*1e15
    p = nextprime(NUMBER_BEFORE)
    g = find_generator_of_group_z(p)
    print(f"{p=}; {g=};")
    list_of_sets = []

    for _ in tqdm(range(NUMBER_OF_SETS)):
        list_of_sets.append(
            generate_random_list_of_sets(NUMBER_OF_ELEMENTS, p, g))

    mean_std_cvar_uniform_randomness(list_of_sets)

    with open("NIST-Statistical-Test-Suite/sts/data/data_bm_dlog.bnr",
              "wb") as f:
        f.write("".join(list_of_sets).encode())

    number_of_elements_list = [1_000, 5_000, 10_000, 50_000,
                               100_000, 500_000, 1_000_000]
    elements_for_mean = 10

    standart_gen_time = []
    custom_gen_time = []
    for number_of_elements in tqdm(number_of_elements_list):
        standart_gen_time.append(
            np.mean([generate_random_list_of_sets_timed(number_of_elements,
                                                        p, g)
                     for _ in range(elements_for_mean)]))
        custom_gen_time.append(
            np.mean([standart_genration(number_of_elements)
                     for _ in range(elements_for_mean)]))

    fig, ax = plt.subplots()
    plt.plot(np.log10(number_of_elements_list), standart_gen_time, "-o",
             label="Созданный генератор")
    plt.plot(np.log10(number_of_elements_list), custom_gen_time, "-o",
             label="Стандартный генератор")
    plt.legend()

    ax.set_title("Зависимость времени генерации от "
                 "кол-ва создаваемых случайных значений",
                 loc='center', wrap=True)

    plt.xlabel("log_10 от кол-ва элементов")
    plt.ylabel("Время работы (секунды)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("Зависимость кол-ва создаваемых"
                " случайных значений от времени.png")
