from .pa_rb_env import (
    PAEnv,
    Node
)
import numpy as np
from pathlib import Path

log2 = np.log2

cues = {
    0: Node(0.1, 0, 'cue'),
    1: Node(-0.1, 0, 'cue'),
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


def test_init_pos():
    """test position constraint"""
    env = PAEnv(n_level=4)

    def dis(node, target):
        return np.sqrt(
            (node.x - target.x) ** 2 +
            (node.y - target.y) ** 2
        )
    # test bs cues
    assert all(
        env.r_bs <= dis(usr, env.station) <= env.R_bs
        for usr in env.cues.values()
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
    env = PAEnv(n_level=4, n_pair=2, m_cue=2)
    env.cues = cues
    env.devices = devices
    env.init_path_loss()
    distance_matrix = env.distance_matrix
    target_dis = np.array(
        [
            [0.1, 1.1, np.sqrt(0.26), np.sqrt(0.26), 0.5],
            [1.1, 0.1, np.sqrt(0.26), np.sqrt(0.26), 0.5],
            [0.6, 0.6, 0.1, 0.1, 0.503],
            [np.sqrt(0.37), np.sqrt(0.37), 0.503, 0.2, 0.1],
            [np.sqrt(0.37), np.sqrt(0.37), 0.2, 0.503, 0.1],
        ]
    )
    assert equal(distance_matrix, target_dis)


def test_get_recv_powers():
    """test get_recv_powers"""
    env = PAEnv(n_level=4, n_pair=2, m_cue=1)
    power = np.array([
        [0.01, 0],
        [0, 0.01],
        [0.1, 0],
        [0, 0.1],
    ])
    emit_powers = np.tile(np.expand_dims(power, axis=1),
                          (1, env.n_channel, 1))
    fading = np.array([
        [1.1e-2, 1.2e-2, 1.3e-2, 1.4e-2],
        [2.1e-2, 2.2e-2, 2.3e-2, 2.4e-2],
        [3.1e-2, 3.2e-2, 3.3e-2, 3.4e-2],
        [4.1e-2, 4.2e-2, 4.3e-2, 4.4e-2],
    ])
    recv_powers = env.get_recv_powers(emit_powers, fading)
    target_recv_powers = np.array([
        [[1.1e-4, 0], [1.2e-4, 0], [1.3e-4, 0], [1.4e-4, 0]],
        [[0, 2.1e-4], [0, 2.2e-4], [0, 2.3e-4], [0, 2.4e-4]],
        [[3.1e-3, 0], [3.2e-3, 0], [3.3e-3, 0], [3.4e-3, 0]],
        [[0, 4.1e-3], [0, 4.2e-3], [0, 4.3e-3], [0, 4.4e-3]],
    ])
    assert equal(recv_powers, target_recv_powers)


def test_get_rates():
    """test get_rates"""
    env = PAEnv(n_level=4, n_pair=2, m_cue=1)
    recv_powers = np.array([
        [[1.1e-4, 0], [1.2e-4, 0], [1.3e-4, 0], [1.4e-4, 0]],
        [[0, 2.1e-4], [0, 2.2e-4], [0, 2.3e-4], [0, 2.4e-4]],
        [[3.1e-3, 0], [3.2e-3, 0], [3.3e-3, 0], [3.4e-3, 0]],
        [[0, 4.1e-3], [0, 4.2e-3], [0, 4.3e-3], [0, 4.4e-3]],
    ])
    rates = env.get_rates(recv_powers)
    _rate = np.array([
        log2(1+1.1/31), log2(1+2.2/42), log2(1+33/1.3), log2(1+44/2.4)
    ])
    target_rates = (_rate * np.ones((env.n_channel, env.n_channel))).T
    assert equal(rates, target_rates)


def test_get_indices():
    """test get_indices"""
    env = PAEnv(n_level=4, n_pair=2, m_cue=1, sorter="recv_power",
                m_state=2)
    power = np.array([
        [0.01, 0],
        [0, 0.01],
        [0.1, 0],
        [0, 0.1],
    ])
    emit_powers = np.tile(np.expand_dims(power, axis=1),
                          (1, env.n_channel, 1))
    fading = np.array([
        [1.1e-2, 1.2e-2, 1.3e-2, 1.4e-2],
        [2.1e-2, 2.2e-2, 2.3e-2, 2.4e-2],
        [3.1e-2, 3.2e-2, 3.3e-2, 3.4e-2],
        [4.1e-2, 4.2e-2, 4.3e-2, 4.4e-2],
    ])
    recv_powers = env.get_recv_powers(emit_powers, fading)
    rates = env.get_rates(recv_powers)
    metrics = emit_powers, recv_powers, rates, fading
    # rx_indice don't need test
    tx_indice, rx_indice = env.get_indices(*metrics)
    target_tx_indice = np.array([
        [3, 3, 3, 2],
        [0, 1, 2, 3]
    ])
    assert equal(tx_indice, target_tx_indice)


def test_get_rewards():
    env = PAEnv(n_level=4, n_pair=2, m_cue=1, sorter="recv_power",
                m_state=2)
    power = np.array([
        [0.01, 0],
        [0, 0.01],
        [0.1, 0],
        [0, 0.1],
    ])
    emit_powers = np.tile(np.expand_dims(power, axis=1),
                          (1, env.n_channel, 1))
    fading = np.array([
        [1.1e-2, 1.2e-2, 1.3e-2, 1.4e-2],
        [2.1e-2, 2.2e-2, 2.3e-2, 2.4e-2],
        [3.1e-2, 3.2e-2, 3.3e-2, 3.4e-2],
        [4.1e-2, 4.2e-2, 4.3e-2, 4.4e-2],
    ])
    recv_powers = env.get_recv_powers(emit_powers, fading)
    rates = env.get_rates(recv_powers)

    metrics = emit_powers, recv_powers, rates, fading
    indices = env.get_indices(*metrics)
    rewards = env.get_rewards(rates, indices)
    target_rewards = np.array([
        log2(1+1.1/31) + log2(1+44/2.4),
        log2(1+2.2/42) + log2(1+44/2.4),
        log2(1+33/1.3) + log2(1+44/2.4),
        log2(1+44/2.4) + log2(1+33/1.3),
    ])[:2]

    assert equal(rewards, target_rewards)


def test_get_states():
    # test m_state
    env = PAEnv(n_level=4, n_pair=2, m_cue=1,
                m_state=8, metrics=['emit_power', 'recv_power', 'rate'],
                sorter='recv_power')
    assert env.m_state == 4

    env = PAEnv(n_level=4, n_pair=2, m_cue=1,
                m_state=2, metrics=['emit_power', 'recv_power', 'rate'],
                sorter='recv_power')
    power = np.array([
        [0.01, 0],
        [0, 0.01],
        [0.1, 0],
        [0, 0.1],
    ])
    emit_powers = np.tile(np.expand_dims(power, axis=1),
                          (1, env.n_channel, 1))
    fading = np.array([
        [1.1e-2, 1.2e-2, 1.3e-2, 1.4e-2],
        [2.1e-2, 2.2e-2, 2.3e-2, 2.4e-2],
        [3.1e-2, 3.2e-2, 3.3e-2, 3.4e-2],
        [4.1e-2, 4.2e-2, 4.3e-2, 4.4e-2],
    ])
    recv_powers = env.get_recv_powers(emit_powers, fading)
    rates = env.get_rates(recv_powers)

    metrics = emit_powers, recv_powers, rates, fading
    indices = env.get_indices(*metrics)
    states = env.get_states(*metrics, indices=indices)
    _recv = np.array([
        [[1.1e-4, 0], [1.2e-4, 0], [1.3e-4, 0], [1.4e-4, 0]],
        [[0, 2.1e-4], [0, 2.2e-4], [0, 2.3e-4], [0, 2.4e-4]],
        [[3.1e-3, 0], [3.2e-3, 0], [3.3e-3, 0], [3.4e-3, 0]],
        [[0, 4.1e-3], [0, 4.2e-3], [0, 4.3e-3], [0, 4.4e-3]],
    ])
    _rate = np.array([
        log2(1+1.1/31), log2(1+2.2/42), log2(1+33/1.3), log2(1+44/2.4)
    ])
    target_states = np.array([
        np.concatenate([power[3],power[0],_recv[3][0],_recv[0][0],[_rate[3], _rate[0]]]),
        np.concatenate([power[3],power[1],_recv[3][1],_recv[1][1],[_rate[3], _rate[1]]]),
        np.concatenate([power[3],power[2],_recv[3][2],_recv[2][2],[_rate[3], _rate[2]]]),
        np.concatenate([power[2],power[3],_recv[2][3],_recv[3][3],[_rate[2], _rate[3]]]),
    ])[:2]
    assert equal(states, target_states)


def test_sorter():
    # now only recv_power can be sorter
    pass


def test_seed():
    env = PAEnv(n_level=4, m_cue=1, seed=123)
    # this is func in PAEnv to random pos

    def random_point(min_r, radius, ox=0, oy=0):
        theta = np.random.random() * 2 * np.pi
        r = np.random.uniform(min_r, radius**2)
        x, y = np.cos(theta) * np.sqrt(r), np.sin(theta) * np.sqrt(r)
        return ox + x, oy + y
    np.random.seed(123)
    target_x, target_y = random_point(env.r_bs, env.R_bs)
    usr = env.cues[0]
    assert all((target_x == usr.x, target_y == usr.y))


def test_action():
    env = PAEnv(n_level=10, seed=799345)
    n_actions = env.n_actions
    n_channel, n_pair = env.n_channel, env.n_pair
    # normal
    env.reset()
    np.random.seed(799345)
    action = np.random.randint(0, n_actions, (n_channel, ))
    s_, r, d, i = env.step(action, unit='dBm')
    assert i['rate'] == 3.4741923099965257
    # only D2D actions is enough
    env.reset()
    np.random.seed(799345)
    action = np.random.randint(0, n_actions, (n_pair, ))
    s_, r, d, i = env.step(action, unit='dBm')
    assert i['rate'] == 3.4741923099965257 
    # other action dim raises error
    env.reset()
    np.random.seed(799345)
    action = np.random.randint(0, n_actions, (n_pair - 1, ))
    try:
        s_, r, d, i = env.step(action, unit='dBm')
    except ValueError as e:
        msg = f"length of action should be n_channel({env.n_channel})" \
            f" or n_pair({n_pair}), but is {len(action)}"
        assert e.args[0] == msg

    env.reset()
    np.random.seed(799345)
    action = np.random.randint(0, n_actions, (n_channel, ))
    s_, r, d, i = env.step(action, unit='mW')
    assert i['rate'] == 3.4928823957853856
    # TODO  add test of continuous action


def test_step():
    env = PAEnv(n_level=10)
    n_actions, n_states = env.n_actions, env.n_states
    assert n_actions == 40
    assert n_states == 304
    env.reset()
    action = env.sample()
    env.step(action, unit='dBm')
    # action = env.sample()
    action = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0])
    env.step(action, unit='mW')
    action = env.sample()
    try:
        env.step(action, unit='xx')
    except ValueError as e:
        msg = f"unit should in ['dBm', 'mW'], but is xx"
        assert e.args[0] == msg
    fig: Path() = env.render()
    if fig.exists():
        fig.unlink()


if __name__ == '__main__':
    test_action()
