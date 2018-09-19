ix = 0
residual = 0
old_block = None
while True:
    chuck = chucks[ix]
    for i in range(residual, len(chuck), batch_size):
        end_slice = i + batch_size
        if (end_slice <= len(chuck)) & (old_block is None):
            yield data[i : end_slice]
        elif old_block is not None:
            new_block = np.concatenate([old_block, data[i : end_slice]])
            old_block = None
            yield new_block
        else:
            residual = len(chuck) - end_slice
            old_block = data[i : end_slice]

    ix = (ix + 1) % len(chucks)
