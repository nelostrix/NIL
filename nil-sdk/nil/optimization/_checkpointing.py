from nil.runtime import require
torch=require('torch','torch')


class ActivationCheckpoint(torch.nn.Module):
    def __init__(self,sequence,segments):
        super().__init__(); self.sequence=sequence; self.segments=segments
    def forward(self,inputs):
        if not self.training or not torch.is_grad_enabled(): return self.sequence(inputs)
        from torch.utils.checkpoint import checkpoint_sequential
        return checkpoint_sequential(self.sequence,self.segments,inputs,use_reentrant=False,preserve_rng_state=True)
