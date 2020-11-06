from .pa_env import (
    PAEnv,
    Node
)
import numpy as np
from pathlib import Path

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


def equal(unit, target):
    tolerance = 1e-6 * np.ones_like(target)
    return (np.abs(unit - target) < tolerance).all()


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
    for cluster in env.devices.values():
        t_device, r_devices = cluster['t_device'], cluster['r_devices']

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
            [np.sqrt(0.26), np.sqrt(0.26), 0.1, 0.1],
            [np.sqrt(0.26), np.sqrt(0.26), 0.1, 0.1],
        ]
    )
    assert equal(distance_matrix, target_dis)


def test_cal_rate():
    """test rate calc"""
    env = PAEnv(n_levels=4, n_t_devices=2, m_r_devices=1, m_usrs=2)
    power = [0.01, 0.01, 0.1, 0.1]
    fading = np.array([
        [1e-1, 1e-3, 1e-2, 1e-2],
        [1e-3, 1e-1, 1e-2, 1e-2],
        [1e-2, 1e-2, 1e-2, 1e-2],
        [1e-2, 1e-2, 1e-2, 1e-2],
    ])
    target_rate = np.array([0.58256799, 0.58256799, 0.87446912, 0.87446912])
    assert equal(env.cal_rate(power, fading), target_rate)


def test_get_state():
    env = PAEnv(n_levels=4, n_t_devices=4, m_r_devices=1, m_usrs=0)
    power = [0.01, 0.02, 0.03, 0.04]
    fading = np.array([
        [1.1e-1, 1.2e-3, 1.3e-2, 1.4e-2],
        [2.1e-3, 2.2e-1, 2.3e-2, 2.4e-2],
        [3.1e-2, 3.2e-2, 3.3e-2, 3.4e-2],
        [4.1e-2, 4.2e-2, 4.3e-2, 4.4e-2],
    ])
    rate = env.cal_rate(power, fading)
    # test error
    try:
        env.m_state = 8
        env.get_state(power, rate, fading)
    except Exception as e:
        assert e.__class__ == ValueError
        assert e.args[0] == 'm_state should be less than n_recvs(4)'\
            ', but was 8'

    # test value
    env.m_state = 2
    target_state = np.array(
        [[0.01, 0.03, 0.04,
          1.09042222, 0.51457317, 0.75950816,
          0.16115479, 0.1728366],
         [0.02, 0.03, 0.04,
          1.86122244, 0.51457317, 0.75950816,
          0.14345279, 0.14937762],
         [0.03, 0.02, 0.04,
          0.51457317, 1.86122244, 0.75950816,
          0.97797369, 1.02169507],
         [0.04, 0.02, 0.03,
          0.75950816, 1.86122244, 0.51457317,
          0.96683314, 0.98351188]]
    )
    assert equal(env.get_state(power, rate, fading), target_state)


def test_metrics():
    # test error
    try:
        env = PAEnv(n_levels=4, n_t_devices=4, m_r_devices=1,
                    m_usrs=0, metrics=["others"])
    except ValueError as e:
        assert e.args[0] == \
            "metrics should in power, rate and fading, but is ['others']"

    power = [0.01, 0.02, 0.03, 0.04]
    fading = np.array([
        [1.1e-1, 1.2e-3, 1.3e-2, 1.4e-2],
        [2.1e-3, 2.2e-1, 2.3e-2, 2.4e-2],
        [3.1e-2, 3.2e-2, 3.3e-2, 3.4e-2],
        [4.1e-2, 4.2e-2, 4.3e-2, 4.4e-2],
    ])
    # test power
    # one test is enough, case total combine metrics is tested in test_state
    env = PAEnv(n_levels=4, n_t_devices=4, m_r_devices=1, m_state=2,
                m_usrs=0, metrics=["power"])
    rate = env.cal_rate(power, fading)

    target_state = np.array(
        [[0.01, 0.03, 0.04, ],
         [0.02, 0.03, 0.04, ],
         [0.03, 0.02, 0.04, ],
         [0.04, 0.02, 0.03, ]]
    )
    assert equal(env.get_state(power, rate, fading), target_state)


def test_sorter():
    # test error
    try:
        env = PAEnv(n_levels=4, n_t_devices=4,
                    m_r_devices=1, m_usrs=0, sorter="others")
    except ValueError as e:
        assert e.args[0] == 'sorter should in power, rate'\
            ' and fading, but is others'

    # test power with power
    power = [0.04, 0.03, 0.02, 0.01]
    fading = np.array([
        [1.1e-1, 1.2e-3, 1.3e-2, 1.4e-2],
        [2.1e-3, 2.2e-1, 2.3e-2, 2.4e-2],
        [3.1e-2, 3.2e-2, 3.3e-2, 3.4e-2],
        [4.1e-2, 4.2e-2, 4.3e-2, 4.4e-2],
    ])
    env = PAEnv(n_levels=4, n_t_devices=4, m_r_devices=1, m_state=2,
                m_usrs=0, sorter="power", metrics=["power"])
    rate = env.cal_rate(power, fading)

    target_state = np.array(
        [[0.04, 0.02, 0.03, ],
         [0.03, 0.02, 0.04, ],
         [0.02, 0.03, 0.04, ],
         [0.01, 0.03, 0.04, ]]
    )
    assert equal(env.get_state(power, rate, fading), target_state)


def test_seed():
    env = PAEnv(n_levels=4, m_usrs=1, seed=123)
    # this is func in PAEnv to random pos

    def random_point(min_r, radius, ox=0, oy=0):
        theta = np.random.random() * 2 * np.pi
        r = np.random.uniform(min_r, radius**2)
        x, y = np.cos(theta) * np.sqrt(r), np.sin(theta) * np.sqrt(r)
        return ox + x, oy + y
    np.random.seed(123)
    target_x, target_y = random_point(env.r_bs, env.R_bs)
    usr = env.users[0]
    assert all((target_x == usr.x, target_y == usr.y))


def test_action():
    env = PAEnv(n_levels=10, seed=799345)
    n_actions = env.n_actions
    n_recvs = env.n_recvs
    # normal
    env.reset()
    np.random.seed(799345)
    action = np.random.randint(0, n_actions, (n_recvs, ))
    s_, r, d, i = env.step(action)
    assert r == 22.252017751938354
    # only D2D actions is enough
    env.reset()
    np.random.seed(799345)
    action = np.random.randint(0, n_actions, (n_recvs - env.m_usr, ))
    s_, r, d, i = env.step(action)
    assert r == 22.252017751938354
    # other action dim raises error
    env.reset()
    np.random.seed(799345)
    action = np.random.randint(0, n_actions, (n_recvs - env.m_usr - 1, ))
    try:
        s_, r, d, i = env.step(action)
    except ValueError as e:
        msg = f"length of power should be n_recvs({env.n_recvs})" \
            f" or n_t*m_r({env.n_t*env.m_r}), but is {len(action)}"
        assert e.args[0] == msg
    # raw
    env.reset()
    np.random.seed(799345)
    action = np.random.randint(0, n_actions, (n_recvs, ))
    raw_power = env.power_levels[action]
    s_, r, d, i = env.step(raw_power, raw=True)
    assert r == 22.252017751938354


def test_step():
    env = PAEnv(n_levels=10)
    n_actions, n_states = env.n_actions, env.n_states
    assert n_actions == 10
    assert n_states == 50
    env.reset()
    action = env.sample()
    env.step(action)
    fig: Path() = env.render()
    if fig.exists():
        fig.unlink()
