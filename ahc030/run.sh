
#cargo run --bin a --features local < tools/in/0000.txt > out
cargo run -r --manifest-path tools/Cargo.toml --bin tester cargo run -r --bin writer < in > out