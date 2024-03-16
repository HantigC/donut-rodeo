class _Slice:
    def __getitem__(self, idx):
        return idx

Slice = _Slice()