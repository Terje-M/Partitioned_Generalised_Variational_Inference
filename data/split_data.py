import numpy as np


def homogenous(x, y, m, dataset_seed):
    """
    Homogenous split of the data into m groups.
    """
    random_state = np.random.get_state()

    if dataset_seed is not None:
        np.random.seed(dataset_seed)

    if m == 1:
        client_data = [{"x": x, "y": y}]
        return client_data

    perm = np.random.permutation(len(x))
    client_data = []
    for i in range(m):
        client_data.append({
            "x": x[perm[i::m]],
            "y": y[perm[i::m]],
        })

    np.random.set_state(random_state)

    return client_data


def inhomogenous1(x, y, m, client_size_factor, class_balance_factor,
                  dataset_seed):
    """
    Splits the data into m groups, half of which are small clients and the
    other half are large clients. Split is based upon the output distribution.
    """
    random_state = np.random.get_state()

    if dataset_seed is not None:
        np.random.seed(dataset_seed)

    if m == 1:
        client_data = [{"x": x, "y": y}]
        return client_data

    if m % 2 != 0:
        raise ValueError('Num clients should be even for nice maths')

    n = x.shape[0]
    small_client_size = int(np.floor((1 - client_size_factor) * n / m))
    big_client_size = int(np.floor((1 + client_size_factor) * n / m))

    class_balance = np.mean(y == 0)

    small_client_class_balance = class_balance + (
                1 - class_balance) * class_balance_factor
    small_client_negative_class_size = int(
        np.floor(small_client_size * small_client_class_balance))
    small_client_positive_class_size = int(
        small_client_size - small_client_negative_class_size)

    if small_client_negative_class_size < 0:
        raise ValueError('Small_client_negative_class_size is negative, '
                         'invalid settings.')
    if small_client_positive_class_size < 0:
        raise ValueError('Small_client_positive_class_size is negative, '
                         'invalid settings.')

    if small_client_negative_class_size * m / 2 > class_balance * n:
        raise ValueError(
            f'Not enough negative class instances to fill the small clients. '
            f'Client size factor:{client_size_factor}, class balance '
            f'factor:{class_balance_factor}')

    if small_client_positive_class_size * m / 2 > (1 - class_balance) * n:
        raise ValueError(
            f'Not enough positive class instances to fill the small clients. '
            f'Client size factor:{client_size_factor}, class balance '
            f'factor:{class_balance_factor}')

    pos_inds = np.where(y > 0)
    zero_inds = np.where(y == 0)

    assert (len(pos_inds[0]) + len(zero_inds[0])) == len(y), \
        "Some indeces missed."

    y_pos = y[pos_inds]
    y_neg = y[zero_inds]

    x_pos = x[pos_inds]
    x_neg = x[zero_inds]

    client_data = []

    # Populate small classes.
    for i in range(int(m / 2)):
        client_x_pos = x_pos[:small_client_positive_class_size]
        x_pos = x_pos[small_client_positive_class_size:]
        client_y_pos = y_pos[:small_client_positive_class_size]
        y_pos = y_pos[small_client_positive_class_size:]

        client_x_neg = x_neg[:small_client_negative_class_size]
        x_neg = x_neg[small_client_negative_class_size:]
        client_y_neg = y_neg[:small_client_negative_class_size]
        y_neg = y_neg[small_client_negative_class_size:]

        client_x = np.concatenate([client_x_pos, client_x_neg])
        client_y = np.concatenate([client_y_pos, client_y_neg])

        shuffle_inds = np.random.permutation(client_x.shape[0])

        client_x = client_x[shuffle_inds, :]
        client_y = client_y[shuffle_inds]

        client_data.append({'x': client_x, 'y': client_y})

    # Recombine remaining data and shuffle.
    x = np.concatenate([x_pos, x_neg])
    y = np.concatenate([y_pos, y_neg])
    shuffle_inds = np.random.permutation(x.shape[0])

    x = x[shuffle_inds]
    y = y[shuffle_inds]

    # Distribute among large clients.
    for i in range(int(m / 2)):
        client_x = x[:big_client_size]
        client_y = y[:big_client_size]

        x = x[big_client_size:]
        y = y[big_client_size:]

        client_data.append({'x': client_x, 'y': client_y})

    np.random.set_state(random_state)

    return client_data



