import ase.io

frames = ase.io.read("data/xyz/mad-test-consistent.xyz", ":")
allowed_Z = {1, 6, 7, 8, 9, 15, 16, 17, 35, 53}

filtered_frames = [
    frame for frame in frames if set(frame.get_atomic_numbers()).issubset(allowed_Z)
]

print(f"kept {len(filtered_frames)} frames out of {len(frames)}")

ase.io.write("data/xyz/mad-test-consistent-organic.xyz", filtered_frames)
