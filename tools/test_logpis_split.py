import torch
from gympn.data import TrajectoryBuffer
from torch_geometric.data import HeteroData

def main():
    buf = TrajectoryBuffer()

    # Build a sample HeteroData state with 2 a_transition nodes and 1 postpone node
    g = HeteroData()
    g['a_transition'].x = torch.randn(2, 4)
    g['postpone'].x = torch.randn(1, 4)
    state = {'graph': g}

    # Create logpis corresponding to a_transition + postpone (length 3)
    logpis = torch.tensor([-0.5, -1.2, -0.1], dtype=torch.float32)

    # Store two steps
    buf.store(state, action=0, reward=0.0, logprob=-0.5, value=0.1, logpis=logpis)
    buf.store(state, action=1, reward=0.0, logprob=-1.0, value=0.2, logpis=logpis)

    # finish episode to compute advantages
    buf.finish()

    # Retrieve loader
    loader = buf.get(batch_size=2, normalize_advantages=False, normalize_returns=False, sort=False, drop_remainder=False)

    for batch in loader:
        print('Batch node types:', list(batch.x_dict.keys()))
        if 'a_transition' in batch.x_dict:
            a_lp = getattr(batch['a_transition'], 'logpis', None)
            print('a_transition.logpis:', a_lp)
        if 'postpone' in batch.x_dict:
            p_lp = getattr(batch['postpone'], 'logpis', None)
            print('postpone.logpis:', p_lp)
        print('whole-step g.logpis:', getattr(batch, 'logpis', None))

if __name__ == '__main__':
    main()

