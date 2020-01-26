import numpy as np
import numpy.fft as fft


def generate_per_sequence_fft(data: np.ndarray, axis: int = 1) -> np.ndarray:
    """
    Performs a FFT along an axis of the dataset.
    """
    data_fft = fft.rfft(data, axis=axis)
    return data_fft


def squared_L2_norm_complex_cumsum(sample: np.ndarray, len_x: int, axis: int) -> np.ndarray:
    """
    Calculates the squared L2 norm of a complex valued vector.
    :param sample: FFT dataset.
    :param len_x: Length of the original dataset, i.e. before applying the FFT.
    :return:
    """
    total = np.cumsum(sample.real ** 2 + sample.imag ** 2, axis=axis)
    return total / (len_x ** 2)


def get_k_fft_by_percentage_energy_above_mean(data: np.array, energy: float,
                                              return_complex: bool = False) -> tuple:
    """
    Truncate the Fourier transform and reconstruct the approximate series using a minimum
    energy bound and always including the first term of the Fourier transform.
    :param return_complex: Flag parameter to return the FFT data in the frequency domain.
    :param data: Dataset to use
    :param energy: Minimum energy threshold
    :return: reconstructed dataset, average number k of terms included in the truncated Fourier transform
    """
    # Non-normalized FFT
    fft_real = generate_per_sequence_fft(data)

    # Calculate the energy cumsum for each term
    sum_energy = (
            2 * squared_L2_norm_complex_cumsum(fft_real, data.shape[1], axis=1))

    mean_terms = sum_energy[:, 0, :]

    sum_energy = sum_energy - mean_terms[:, None, :]
    max_cum = sum_energy[:, -1, :]

    sum_energy = sum_energy / max_cum[:, None, :]

    k_thresh = np.argmax(sum_energy >= energy, axis=1)
    avg_k = np.mean(k_thresh)
    # print("Average k:", avg_k)
    # print("Average communication saving:", 1 - np.mean(k_thresh) * 2 / data.shape[1])

    k_filter = sum_energy >= energy
    fft_real[k_filter] = 0

    # print("Data shape in trunc func:", data.shape)
    # print("FFT shape in trunc func:", fft_real.shape)

    if return_complex:
        return fft_real, avg_k

    k_real_recon = fft.irfft(fft_real, n=data.shape[1], axis=1)

    # plt.figure()
    # plt.title('k:' + str(k_thresh[0, 0]))
    # plt.plot(k_real_recon[0, :, 0], label='approx')
    # plt.plot(data[0, :, 0], label='real')
    # plt.legend(loc='upper right')
    # plt.show()

    return k_real_recon, avg_k


def get_k_fft_by_max_RMSE(data: np.array, max_err: float, return_complex: bool = False):
    # Non-normalized FFT
    fft_real = generate_per_sequence_fft(data)

    # Calculate the energy cumsum for each term
    sum_energy = (
            2 * squared_L2_norm_complex_cumsum(fft_real, data.shape[1], axis=1))

    # Take the root since we are dealing with RMSE
    max_cum = sum_energy[:, -1, :]

    sum_energy = max_cum[:, None, :] - sum_energy

    error_caused = np.sqrt(sum_energy)

    k_thresh = np.argmax(error_caused < max_err, axis=1)
    avg_k = np.mean(k_thresh)

    print("error caused:", error_caused[0, :, 0])
    k_filter = error_caused < max_err
    fft_real[k_filter] = 0

    if return_complex:
        return fft_real, avg_k

    k_real_recon = fft.irfft(fft_real, n=data.shape[1], axis=1)

    return k_real_recon, avg_k
