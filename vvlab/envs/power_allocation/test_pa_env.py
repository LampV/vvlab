from pa_env import (
    PAEnv,
    Node
)
import numpy as np

users = {
    0: Node(0.1, 0, 'user'),
    1: Node(-0.1, 0, 'user'),
}
devices = {
    0: {
        't_device': Node(0, 0.5, 't_device'),
        'r_devices': {
            0: Node(0, 0.6, 'r_device')
        }
    },
    1: {
        't_device': Node(0, -0.5, 't_device'),
        'r_devices': {
            0: Node(0, -0.6, 'r_device')
        }
    }
}


def test_init_power_level():
    """test transmit powers"""
    env = PAEnv(n_levels=4, min_power=10, max_power=30)
    power_levels = env.power_levels
    assert all(power_levels == [0., 0.01, 0.1, 1.])


def test_init_pos():
    """test position constraint"""
    env = PAEnv(n_levels=4)

    def dis(node, target):
        return np.sqrt(
            (node.x - target.x) ** 2 +
            (node.y - target.y) ** 2
        )
    # test bs users
    assert all(
        env.r_bs <= dis(usr, env.station) <= env.R_bs
        for usr in env.users.values()
    )

    # test devices
    for pair in env.devices.values():
        t_device, r_devices = pair['t_device'], pair['r_devices']

        assert env.r_bs <= dis(t_device, env.station) <= (
            env.R_bs - env.R_dev)
        assert all(
            env.r_dev <= dis(r_device, t_device) <= env.R_dev
            for r_device in r_devices.values()
        )


def test_jakes():
    # TODO test stastic features of jakes
    # target_std, target_mean = 0.429, 1.253  # Rayleigh Distribution

    # x_len, y_len, Ns = H_set.shape
    # h_std = np.mean([
    #     H_set[x, y, :].std()
    #     for x in range(x_len)
    #     for y in range(y_len)
    # ])
    # assert (h_std - target_std) / target_std < 0.1

    # h_mean = np.mean([
    #     H_set[x, y, :].mean()
    #     for x in range(x_len)
    #     for y in range(y_len)
    # ])
    # assert (h_mean - target_mean) / target_mean < 0.05
    pass


def test_init_path_loss():
    """test distance, since lognormal is random"""
    env = PAEnv(n_levels=4, n_t_devices=2, m_r_devices=1, m_usrs=2)
    env.users = users
    env.devices = devices
    env.init_path_loss()
    distance_matrix = env.distance_matrix
    target_dis = np.array(
        [
            [0.1, 1.1, 0.6, 0.6],
            [1.1, 0.1, 0.6, 0.6],
            [np.sqrt(0.26), np.sqrt(0.6), 0.1, 0.1],
            [np.sqrt(0.26), np.sqrt(0.6), 0.1, 0.1],
        ]
    )
    tolerance = 1e-6 * np.ones((env.n_recvs, env.n_recvs))
    assert (distance_matrix - target_dis < tolerance).all()


def test_cal_rate():
    """test rate calc"""
    env = PAEnv(n_levels=4, n_t_devices=2, m_r_devices=1, m_usrs=2)
    power = [0.01, 0.01, 0.1, 0.1]
    loss = np.array([
        [1e-1, 1e-3, 1e-2, 1e-2],
        [1e-3, 1e-1, 1e-2, 1e-2],
        [1e-2, 1e-2, 1e-2, 1e-2],
        [1e-2, 1e-2, 1e-2, 1e-2],
    ])
    target_rate = np.array([0.58256799, 0.58256799, 0.87446912, 0.87446912])
    tolerance = 1e-6 * np.ones((env.n_recvs, env.n_recvs))
    assert (env.cal_rate(power, loss) - target_rate < tolerance).all()


def test_get_state():
    env = PAEnv(n_levels=4, n_t_devices=2, m_r_devices=1, m_usrs=2)
    power = [0.01, 0.02, 0.03, 0.04]
    loss = np.array([
        [1.1e-1, 1.2e-3, 1.3e-2, 1.4e-2],
        [2.1e-3, 2.2e-1, 2.3e-2, 2.4e-2],
        [3.1e-2, 3.2e-2, 3.3e-2, 3.4e-2],
        [4.1e-2, 4.2e-2, 4.3e-2, 4.4e-2],
    ])
    rate = env.cal_rate(power, loss)
    # test error
    try:
        env.m_state = 8
        state = env.get_state(rate, power, loss)
    except Exception as e:
        assert e.__class__ == ValueError
        assert e.args[0] == 'm_state(8) cannot be greater than n_recvs(4)'

    # test value
    env.m_state = 2
    target_state = np.array(
        [[1.09042222, 0.51457317, 0.75950816,
          0.01, 0.03, 0.04,
          0.16115479, 0.1728366],
         [1.86122244, 0.51457317, 0.75950816,
          0.02, 0.03, 0.04,
          0.14345279, 0.14937762],
         [0.51457317, 1.86122244, 0.75950816,
          0.03, 0.02, 0.04,
          0.97797369, 1.02169507],
         [0.75950816, 1.86122244, 0.51457317,
          0.04, 0.02, 0.03,
          0.96683314, 0.98351188]]
    )
    tolerance = 1e-6 * np.ones((env.n_recvs, env.m_state * 3 + 2))
    assert (env.get_state(rate, power, loss) -
            target_state < tolerance).all()
