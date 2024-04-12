#![allow(non_snake_case)]
#![allow(unused_imports)]
#![allow(unused_macros)]
#![allow(clippy::comparison_chain)]
#![allow(clippy::nonminimal_bool)]
#![allow(clippy::neg_multiply)]
#![allow(clippy::type_complexity)]
#![allow(clippy::needless_range_loop)]
#![allow(dead_code)]

use std::{
    cmp::Reverse,
    collections::{BinaryHeap, HashMap, VecDeque},
};

use itertools::Itertools;
use proconio::{fastout, input, marker::Chars};
use rand::prelude::*;
use rustc_hash::{FxHashMap, FxHashSet};

// Potential
// t=0: 6984586
// t=1: 8534380
// t=2: 9111393
// t=3: 7862334
// t=4: 7487920
// t=5: 8531081
// t=6: 7125290
// t=7: 11006891
const PRE: [&str; 9] = ["47 46 45 44 100 99 98 95 93 97 48 49 50 51 77 81 84 86 89 96 59 56 53 54 76 79 82 85 87 94 60 57 55 58 73 75 78 83 88 92 43 42 41 63 68 72 74 80 90 91 40 39 38 61 64 67 70 71 2 1 37 36 35 52 62 65 66 69 4 3 33 32 31 34 23 19 16 11 7 5 30 29 27 25 21 18 15 12 9 6 28 26 24 22 20 17 14 13 10 8",
"91 89 88 85 5 6 7 8 10 9 92 90 87 86 4 3 2 17 12 11 94 93 66 84 83 82 1 18 13 14 96 95 67 76 80 81 20 19 16 15 100 97 68 72 79 78 27 21 22 23 99 98 65 73 75 77 28 25 26 24 63 64 62 71 74 40 32 30 29 31 61 60 59 70 69 39 37 38 34 33 58 57 56 55 52 49 41 43 45 35 50 51 53 54 48 47 44 42 46 36",
"78 80 82 84 87 90 89 86 77 76 75 73 71 69 36 79 81 83 85 88 92 93 91 74 72 70 68 67 65 37 151 148 147 101 99 96 95 94 66 64 63 62 61 60 38 150 146 144 105 103 
100 98 97 59 58 57 56 55 54 39 149 143 141 110 108 106 104 102 50 51 53 52 49 46 41 152 137 132 121 114 111 109 107 42 43 47 48 45 44 40 154 133 131 130 116 115 113 112 33 30 22 19 20 28 35 155 136 135 134 120 119 118 117 23 21 18 16 14 32 34 156 138 140 139 125 124 123 122 17 15 13 11 9 29 31 157 153 145 142 129 128 127 126 12 10 8 5 3 26 27 159 158 160 163 183 182 178 177 7 6 4 2 1 24 25 161 162 164 167 184 186 190 194 198 202 206 210 214 224 225 165 166 168 173 180 187 
191 195 199 203 207 211 215 222 223 169 170 171 175 181 188 192 196 200 204 208 212 216 219 221 172 174 176 179 185 189 193 197 201 205 209 213 217 218 220",
"1 4 9 18 32 48 63 78 110 126 134 141 152 161 166 171 2 5 11 19 33 47 60 66 125 129 136 144 156 164 170 174 3 7 13 21 34 46 57 62 130 132 139 149 159 169 176 180 6 8 14 23 35 45 55 61 135 137 143 154 163 173 181 183 10 12 17 26 36 44 53 59 138 140 148 158 168 178 185 187 15 16 20 28 37 43 51 56 142 145 153 162 172 182 188 190 25 22 24 30 38 42 50 54 147 151 157 165 175 184 189 193 40 29 27 31 39 41 49 52 150 155 160 167 177 186 192 198 58 65 71 80 91 97 102 107 204 208 216 
218 226 230 228 217 64 67 73 82 92 100 106 111 203 207 215 219 227 233 235 232 68 69 75 85 95 104 112 116 201 206 214 220 229 237 241 242 70 72 79 89 99 109 117 119 199 205 213 221 231 240 245 247 74 76 84 94 103 114 120 122 196 202 212 222 234 243 249 251 77 81 88 98 108 118 124 127 195 200 211 223 236 244 250 254 83 87 93 101 113 121 128 133 191 197 210 224 238 246 252 255 86 90 96 105 115 123 131 146 179 194 209 225 239 248 253 256",
"2 3 5 20 29 40 54 72 106 110 117 144 180 193 200 215 224 233 234 1 6 10 18 30 41 52 86 101 111 126 148 167 198 206 217 225 232 236 4 8 13 19 31 46 62 81 100 113 132 153 172 192 207 220 229 237 242 7 9 15 21 36 51 69 84 105 123 141 160 177 188 211 226 231 244 248 11 12 16 24 37 55 75 102 115 140 151 170 184 197 216 228 238 249 254 14 23 27 35 38 57 63 114 149 156 179 185 194 196 230 243 255 259 262 17 22 26 39 45 58 67 129 143 157 195 199 208 223 251 257 264 265 270 25 28 32 43 53 66 78 134 152 166 202 209 219 239 250 261 278 287 280 33 34 44 50 65 77 103 124 165 178 201 218 260 263 281 289 297 302 308 42 48 49 64 68 71 128 138 164 182 204 241 258 271 288 294 304 309 311 47 56 59 61 137 139 146 163 173 222 227 247 266 274 301 310 313 315 317 74 70 73 60 136 135 158 171 191 214 235 253 
272 282 300 312 318 319 320 76 80 88 104 118 133 150 169 203 212 246 256 283 291 326 321 325 327 324 79 83 92 107 119 131 147 187 205 221 245 269 286 296 330 333 334 331 361 82 85 93 116 122 154 162 186 213 277 284 285 295 307 328 337 342 350 357 87 90 94 121 127 145 176 183 240 267 290 292 299 305 336 338 347 352 356 89 91 98 109 125 159 174 181 273 279 293 306 329 335 339 341 348 354 358 96 97 99 130 142 161 175 189 268 275 303 316 322 344 343 346 349 353 360 95 108 112 
120 155 168 190 210 252 276 298 314 323 332 340 345 351 355 359",
"1 23 24 65 67 68 73 84 92 98 105 112 118 124 129 131 123 117 114 111 2 26 27 61 66 69 75 86 93 99 107 115 121 127 135 145 169 205 219 225 3 29 30 56 63 70 78 88 94 102 113 119 126 132 141 148 158 223 227 228 8 28 32 44 62 72 82 90 96 100 125 128 134 140 146 151 157 237 235 230 16 22 25 31 71 79 89 95 101 103 136 138 
143 149 155 161 164 248 243 245 17 18 19 20 80 87 97 104 109 110 144 147 153 162 166 170 174 265 273 276 15 13 12 11 85 91 108 116 120 122 154 159 167 176 179 
181 183 270 274 275 14 9 7 6 81 83 130 133 137 139 163 165 184 188 190 194 196 269 271 272 21 10 5 4 76 77 150 152 156 160 168 171 199 202 206 212 216 263 266 
267 38 47 54 57 298 297 172 173 175 177 178 180 211 214 218 224 234 252 260 264 45 51 55 60 295 293 192 191 189 186 185 187 215 221 226 232 240 249 257 261 49 
53 58 64 291 289 220 217 209 195 193 197 208 229 233 239 244 250 255 258 48 52 59 74 286 277 253 241 231 198 200 204 210 236 238 242 246 251 254 256 43 46 50 106 284 279 268 259 247 201 203 207 213 400 399 397 392 393 396 398 40 41 42 142 282 283 280 281 290 335 339 344 348 363 367 374 384 389 394 395 36 37 39 182 278 285 288 292 296 334 338 343 346 360 365 371 380 386 390 391 33 34 35 222 262 287 294 299 302 331 336 342 347 357 362 368 376 382 387 388 327 322 318 314 308 
300 301 305 312 325 333 341 349 355 359 366 372 377 383 385 328 323 319 315 309 303 304 310 317 326 332 340 350 354 358 364 369 373 379 381 329 324 320 316 311 306 307 313 321 330 337 345 351 352 353 356 361 370 375 378",
"119 114 106 97 89 79 70 62 53 46 37 31 25 20 14 11 7 4 2 1 126 121 113 104 94 85 75 66 57 49 41 35 27 22 17 13 9 6 5 3 137 132 125 116 105 95 84 74 65 55 48 40 34 29 23 18 15 12 10 8 149 145 140 130 120 109 96 87 76 68 58 50 43 38 32 26 24 21 19 16 161 158 153 147 139 128 115 102 91 82 72 63 56 52 45 39 36 33 30 28 173 171 168 163 155 146 135 122 110 99 90 81 73 67 60 54 51 47 44 42 189 187 184 179 172 164 154 143 134 123 111 101 93 86 77 71 69 64 61 59 208 206 202 198 192 183 175 166 157 148 138 127 117 107 98 92 88 83 80 78 227 225 221 217 211 204 196 188 180 170 162 151 142 133 124 118 112 108 103 100 250 247 242 238 234 226 
218 209 201 193 185 177 167 159 150 144 141 136 131 129 273 270 265 261 256 248 239 231 222 214 207 199 191 182 174 169 165 160 156 152 301 297 293 287 281 272 262 253 245 237 228 220 212 203 195 190 186 181 178 176 324 321 317 312 307 300 290 279 269 260 251 241 233 224 216 210 205 200 197 194 344 341 337 334 330 322 313 304 294 284 274 263 254 244 236 229 223 219 215 213 361 358 354 351 347 340 332 326 316 308 299 288 277 264 255 249 240 235 232 230 375 371 368 365 362 356 350 343 335 327 318 309 298 286 276 267 258 252 246 243 385 382 380 377 374 369 363 357 349 342 333 325 315 305 295 285 275 266 259 257 393 390 389 386 383 
378 372 366 359 353 346 338 329 320 310 302 291 280 271 268 398 396 394 392 388 384 379 373 367 360 352 345 336 328 319 311 303 292 282 278 400 399 397 395 391 387 381 376 370 364 355 348 339 331 323 314 306 296 289 283",
"363 370 371 321 322 324 328 331 332 383 382 381 385 392 393 397 398 399 31 35 364 368 355 349 348 323 333 334 362 361 367 380 384 391 394 396 395 400 32 34 365 366 360 346 344 341 338 337 343 353 372 376 386 388 64 63 16 17 24 37 374 369 358 359 345 335 339 340 342 347 390 389 387 74 66 67 20 19 25 39 375 373 356 357 327 329 326 330 336 287 291 292 75 73 71 69 23 27 26 40 377 378 354 352 325 318 319 311 301 286 288 289 77 76 70 68 29 30 33 41 295 379 303 351 350 317 320 294 293 284 285 290 80 82 78 60 61 62 38 42 296 298 304 305 312 314 313 275 274 273 83 85 87 86 79 81 59 57 44 45 297 299 302 306 309 315 279 277 235 260 249 84 
120 96 72 65 58 56 52 49 256 300 308 307 310 316 232 231 234 236 238 239 121 105 117 51 50 43 53 48 257 255 252 248 244 237 230 229 191 227 228 220 122 114 116 115 21 36 54 47 258 254 251 250 218 221 226 225 192 194 203 210 132 3 1 18 22 28 55 46 176 175 213 216 219 222 223 245 172 184 148 149 141 5 8 13 12 9 10 90 177 178 214 215 217 224 233 246 173 174 166 159 161 4 2 15 14 7 11 91 180 179 197 211 212 240 241 247 253 165 162 160 163 136 135 101 100 6 95 93 181 182 198 196 209 208 242 243 259 167 168 158 164 138 139 140 99 98 97 94 183 201 200 202 205 204 266 264 262 261 169 154 150 137 133 134 119 104 102 92 185 186 195 199 206 207 267 265 263 171 170 145 146 142 128 123 118 106 103 89 187 190 193 281 280 271 270 269 157 152 151 147 126 127 129 130 113 110 109 88 188 189 283 282 278 276 272 268 156 155 153 144 143 125 124 131 112 111 108 107",
""];

fn main() {
    let start = std::time::Instant::now();
    let time_limit = 1000.0;
    let time_keeper = TimeKeeper::new(time_limit);
    let mut rng = rand_pcg::Pcg64Mcg::new(12345);
    let input = read_input();

    let (init_board, st, goal) = make_init_board_head_and_tail_start(&input);
    let mut state = State::new(
        &input,
        st,
        goal,
        annealing(&input, init_board, &mut rng, &time_keeper),
    );

    // let mut target_board = DynamicMap2d::new_with(0, input.n);
    // for (i, x) in PRE[0]
    //     .split_whitespace()
    //     .map(|v| v.parse::<usize>().unwrap())
    //     .enumerate()
    // {
    //     let pos = Coord::new(i / input.n, i % input.n);
    //     target_board[pos] = x;
    // }

    eprintln!(
        "{}",
        state.best_board.to_2d_vec().iter().flatten().join(" ")
    );

    state.quick_sort(&input);
    state.output();
    let cost = calc_cost(&state.board, &input);
    let score = calc_score(input.cost, cost);
    eprintln!("Cost: {}", cost);
    eprintln!("Score: {}", score);

    #[allow(unused_mut, unused_assignments)]
    let mut elapsed_time = start.elapsed().as_micros() as f64 * 1e-6;
    #[cfg(feature = "local")]
    {
        eprintln!("Local Mode");
        elapsed_time *= 0.55;
    }
    eprintln!("Elapsed: {}", (elapsed_time * 1000.0) as usize);
}

fn make_init_board_head_start(input: &Input) -> DynamicMap2d<i64> {
    let mut board = DynamicMap2d::new_with(0, input.n);
    let mut used = DynamicMap2d::new_with(false, input.n);
    let mut now = 1_i64;
    let mut Q = VecDeque::new();

    let mut cands = vec![];
    for i in 0..input.n {
        for j in 0..input.n {
            let pos = Coord::new(i, j);
            cands.push((input.legal_actions[pos].len(), i, j));
        }
    }
    cands.sort();
    let (_, i, j) = cands[0];
    Q.push_back(Coord::new(i, j));

    while now as usize <= input.n2 {
        while let Some(pos) = Q.pop_front() {
            if used[pos] {
                continue;
            }
            board[pos] = now;
            used[pos] = true;
            for (_, nxt) in input.legal_actions[pos].iter() {
                if !used[*nxt] {
                    Q.push_back(*nxt);
                }
            }
            break;
        }
        now += 1;
    }
    #[cfg(feature = "local")]
    visualizer::vis(input.n, &input.vs, &input.hs, &board.to_2d_vec());
    board
}

fn make_init_board_head_and_tail_start(input: &Input) -> (DynamicMap2d<i64>, Coord, Coord) {
    let mut cands = vec![];
    for i in 0..input.n {
        for j in 0..input.n {
            let pos = Coord::new(i, j);
            cands.push((input.legal_actions[pos].len(), i, j));
        }
    }
    cands.sort();
    let (_, i, j) = cands[0];
    let start = Coord::new(i, j);

    let INF = 1_usize << 60;
    let mut dist = DynamicMap2d::new_with(INF, input.n);
    let mut Q = VecDeque::new();

    dist[start] = 0;
    Q.push_back(start);
    while let Some(pos) = Q.pop_front() {
        for (_, nxt) in input.legal_actions[pos].iter() {
            if dist[pos] + 1 < dist[*nxt] {
                dist[*nxt] = dist[pos] + 1;
                Q.push_back(*nxt);
            }
        }
    }

    let mut cands = vec![];
    for i in 0..input.n {
        for j in 0..input.n {
            let pos = Coord::new(i, j);
            cands.push((dist[pos], Reverse(input.legal_actions[pos].len()), i, j));
        }
    }
    cands.sort();
    cands.reverse();
    let (_, _, i, j) = cands[0];
    let goal = Coord::new(i, j);

    let mut board = DynamicMap2d::new_with(0, input.n);
    let mut used = DynamicMap2d::new_with(false, input.n);
    let mut left = 1_i64;
    let mut right = input.n2 as i64;
    let mut Q1 = VecDeque::new();
    let mut Q2 = VecDeque::new();
    Q1.push_back(start);
    Q2.push_back(goal);
    let mut cnt = 0;

    while left <= right {
        let L = Q1.len();
        for _ in 0..L {
            let pos = Q1.pop_front().unwrap();
            if !used[pos] {
                Q1.push_back(pos);
            }
        }
        let L = Q2.len();
        for _ in 0..L {
            let pos = Q2.pop_front().unwrap();
            if !used[pos] {
                Q2.push_back(pos);
            }
        }

        if Q1.is_empty() {
            Q1 = Q2
                .clone()
                .into_iter()
                .collect_vec()
                .into_iter()
                .rev()
                .collect();
        }
        if Q2.is_empty() {
            Q2 = Q1
                .clone()
                .into_iter()
                .collect_vec()
                .into_iter()
                .rev()
                .collect();
        }

        if cnt % 2 == 0 {
            while let Some(pos) = Q1.pop_front() {
                if used[pos] {
                    continue;
                }
                board[pos] = left;
                used[pos] = true;
                for (_, nxt) in input.legal_actions[pos].iter() {
                    if !used[*nxt] {
                        Q1.push_back(*nxt);
                    }
                }
                break;
            }
            left += 1;
        } else {
            while let Some(pos) = Q2.pop_front() {
                if used[pos] {
                    continue;
                }
                board[pos] = right;
                used[pos] = true;
                for (_, nxt) in input.legal_actions[pos].iter() {
                    if !used[*nxt] {
                        Q2.push_back(*nxt);
                    }
                }
                break;
            }
            right -= 1;
        }
        cnt += 1;
    }
    #[cfg(feature = "local")]
    visualizer::vis(input.n, &input.vs, &input.hs, &board.to_2d_vec());
    (board, start, goal)
}

fn annealing(
    input: &Input,
    board: DynamicMap2d<i64>,
    rng: &mut rand_pcg::Pcg64Mcg,
    time_keeper: &TimeKeeper,
) -> DynamicMap2d<i64> {
    let mut state = AnnealingState::new(board, input);
    // eprintln!("Init cost: {}", state.cost);
    let mut best_state = state.clone();

    let T0 = (4 * input.n.pow(2)) as f64;
    let T1 = 1.0;

    while !time_keeper.isTimeOver() {
        let i1 = rng.gen_range(0..input.n);
        let j1 = rng.gen_range(0..input.n);
        let pos1 = Coord::new(i1, j1);
        let i2 = rng.gen_range(0..input.n);
        let j2 = rng.gen_range(0..input.n);
        let pos2 = Coord::new(i2, j2);
        if pos1 == pos2 {
            continue;
        }
        let diff = state.calc_diff_cost(pos1, pos2, input);
        let temp = T0 + (T1 - T0) * time_keeper.get_time() / time_keeper.time_threshold;
        if diff <= 0 || rng.gen_bool((-diff as f64 / temp).exp()) {
            state.swap(pos1, pos2);
            state.cost += diff;
        }
        if state.cost < best_state.cost {
            best_state = state.clone();
            best_state.cost = state.cost;
        }
    }
    let score = calc_score(input.cost, best_state.cost);
    eprintln!("Potential: {}", score);
    #[cfg(feature = "local")]
    visualizer::vis(input.n, &input.vs, &input.hs, &best_state.board.to_2d_vec());
    best_state.board
}

fn calc_cost(board: &DynamicMap2d<i64>, input: &Input) -> i64 {
    let mut cost = 0;
    for i in 0..input.n {
        for j in 0..input.n {
            let coord = Coord::new(i, j);
            let coord_down = Coord::new(i + 1, j);
            let coord_right = Coord::new(i, j + 1);

            if i + 1 < input.n && input.hs[i][j] == '0' {
                cost += (board[coord] - board[coord_down]).pow(2);
            }
            if j + 1 < input.n && input.vs[i][j] == '0' {
                cost += (board[coord] - board[coord_right]).pow(2);
            }
        }
    }
    cost
}

fn calc_score(init_cost: i64, cost: i64) -> i64 {
    let score = (init_cost as f64).log2() - (cost as f64).log2();
    let mut score = (1e6 * score).round() as i64;
    score = score.max(1);
    score
}

#[derive(Debug, Clone, Default)]
struct State {
    n: usize,
    n2: usize,
    init_pos1: Coord,
    init_pos2: Coord,
    pos1: Coord,
    pos2: Coord,
    board: DynamicMap2d<i64>,
    best_board: DynamicMap2d<i64>,
    actions: Vec<usize>,
}

impl State {
    fn new(input: &Input, pos1: Coord, pos2: Coord, best_board: DynamicMap2d<i64>) -> State {
        State {
            n: input.n,
            n2: input.n2,
            init_pos1: pos1,
            init_pos2: pos2,
            pos1,
            pos2,
            board: input.board.clone(),
            best_board,
            actions: vec![0],
        }
    }
    fn swap(&mut self, pos1: Coord, pos2: Coord) {
        let tmp = self.board[pos1];
        self.board[pos1] = self.board[pos2];
        self.board[pos2] = tmp;
    }
    fn search_different(&self, left: usize, right: usize) -> (usize, DynamicMap2d<u8>) {
        let mut different = DynamicMap2d::new_with(0, self.n);
        let mid = (left + right) / 2;
        let mut cnt = 0;
        for i in 0..self.n {
            for j in 0..self.n {
                let coord = Coord::new(i, j);
                let best_val = self.best_board[coord] as usize - 1;
                let now_val = self.board[coord] as usize - 1;
                if left <= best_val && best_val < mid && mid <= now_val && now_val < right {
                    different[coord] = 1;
                    cnt += 1;
                }
                if left <= now_val && now_val < mid && mid <= best_val && best_val < right {
                    different[coord] = 2;
                    cnt += 1;
                }
            }
        }
        cnt /= 2;
        (cnt, different)
    }
    fn quick_sort(&mut self, input: &Input) {
        let mut Q = vec![];
        let mut next_Q = vec![];
        let mut cnt = 0;
        Q.push((0, self.n2));
        while let Some((left, right)) = Q.pop() {
            if left == right {
                continue;
            }
            self.tsp(input, left, right);
            if self.actions.len() > 3 * 4 * self.n2 {
                break;
            }
            if right - left > 1 {
                let mid = (left + right) / 2;
                next_Q.push((left, mid));
                next_Q.push((mid, right));
            }
            if Q.is_empty() {
                std::mem::swap(&mut Q, &mut next_Q);
                Q.sort();
                if cnt % 2 == 0 {
                    Q.reverse();
                }
                cnt += 1;
            }
        }
    }
    fn tsp(&mut self, input: &Input, left: usize, right: usize) {
        let (cnt, mut different) = self.search_different(left, right);
        for _ in 0..cnt {
            let mut actions1 = self.bfs(input, self.pos1, 1, &mut different);
            let mut actions2 = self.bfs(input, self.pos2, 2, &mut different);
            assert!(!(actions1.is_empty() && actions2.is_empty()));
            while actions1.len() < actions2.len() {
                actions1.push(DIRS_MAP[&'.']);
            }
            while actions1.len() > actions2.len() {
                actions2.push(DIRS_MAP[&'.']);
            }
            for i in 0..actions1.len() {
                self.actions.push(actions1[i]);
                self.pos1 = self.pos1 + DIJ_DIFF[actions1[i]];
                self.actions.push(actions2[i]);
                self.pos2 = self.pos2 + DIJ_DIFF[actions2[i]];
                self.actions.push(0);
            }
            *self.actions.last_mut().unwrap() = 1;
            self.swap(self.pos1, self.pos2);
        }
    }
    fn bfs(
        &self,
        input: &Input,
        st: Coord,
        search: u8,
        different: &mut DynamicMap2d<u8>,
    ) -> Vec<usize> {
        let mut dist: FxHashMap<Coord, usize> = FxHashMap::default();
        dist.insert(st, 0);
        let mut Q = VecDeque::new();
        Q.push_back((st, vec![]));
        while let Some((pos, actions)) = Q.pop_front() {
            for &(dir, nxt) in input.legal_actions[pos].iter() {
                if !dist.contains_key(&nxt) || dist[&pos] + 1 < dist[&nxt] {
                    let mut nxt_actions = actions.clone();
                    nxt_actions.push(dir);
                    if different[nxt] == search {
                        different[nxt] = 0;
                        return nxt_actions;
                    }
                    dist.insert(nxt, dist[&pos] + 1);
                    Q.push_back((nxt, nxt_actions));
                }
            }
        }
        vec![]
    }
    #[fastout]
    fn output(&mut self) {
        self.actions.push(DIRS_MAP[&'.']);
        self.actions.push(DIRS_MAP[&'.']);
        self.actions.truncate(3 * 4 * self.n2);
        println!(
            "{} {} {} {}",
            self.init_pos1.row, self.init_pos1.col, self.init_pos2.row, self.init_pos2.col
        );
        // ターン毎にスワップ、高橋君移動、青木移動のアクションを同時に保存できないので、
        // タプルは使用せずに、1次元配列に保存した。
        // mod3=0: スワップ
        // mod3=1: 高橋君移動
        // mod3=2: 青木君移動
        let mut i = 0;
        while i < self.actions.len() {
            let is_swap = self.actions[i];
            let dir1 = self.actions[i + 1];
            let dir2 = self.actions[i + 2];
            println!("{} {} {}", is_swap, DIRS[dir1], DIRS[dir2]);
            i += 3;
        }
    }
}

#[derive(Debug, Clone, Default)]
struct AnnealingState {
    board: DynamicMap2d<i64>,
    cost: i64,
}

impl AnnealingState {
    fn new(board: DynamicMap2d<i64>, input: &Input) -> AnnealingState {
        let cost = calc_cost(&board, input);
        AnnealingState { board, cost }
    }
    fn swap(&mut self, pos1: Coord, pos2: Coord) {
        let tmp = self.board[pos1];
        self.board[pos1] = self.board[pos2];
        self.board[pos2] = tmp;
    }
    fn calc_diff_cost(&self, pos1: Coord, pos2: Coord, input: &Input) -> i64 {
        // スワップする周辺だけ差分更新
        let mut before = 0;
        for nxt in input.legal_actions[pos1].iter() {
            before += (self.board[pos1] - self.board[nxt.1]).pow(2);
        }
        for nxt in input.legal_actions[pos2].iter() {
            before += (self.board[pos2] - self.board[nxt.1]).pow(2);
        }
        let mut after = 0;
        for nxt in input.legal_actions[pos1].iter() {
            if nxt.1 == pos2 {
                after += (self.board[pos2] - self.board[pos1]).pow(2);
            } else {
                after += (self.board[pos2] - self.board[nxt.1]).pow(2);
            }
        }
        for nxt in input.legal_actions[pos2].iter() {
            if nxt.1 == pos1 {
                after += (self.board[pos1] - self.board[pos2]).pow(2);
            } else {
                after += (self.board[pos1] - self.board[nxt.1]).pow(2);
            }
        }
        after - before
    }
}

const DIRS: [char; 5] = ['U', 'D', 'L', 'R', '.'];
const DIJ: [(usize, usize); 5] = [(!0, 0), (1, 0), (0, !0), (0, 1), (0, 0)];
const DIJ_DIFF: [CoordDiff; 5] = [
    CoordDiff::new(!0, 0),
    CoordDiff::new(1, 0),
    CoordDiff::new(0, !0),
    CoordDiff::new(0, 1),
    CoordDiff::new(0, 0),
];
const DIRS_REVERSE: [usize; 5] = [1, 0, 3, 2, 4];

lazy_static::lazy_static! {
    static ref DIRS_MAP: HashMap<char, usize> = {
        let mut mp = HashMap::new();
        for (i,dir) in DIRS.iter().enumerate() {
            mp.insert(*dir, i);
        }
        mp
    };
}

struct Input {
    t: usize,
    n: usize,
    n2: usize,
    cost: i64,
    board: DynamicMap2d<i64>,
    legal_actions: DynamicMap2d<Vec<(usize, Coord)>>,
    vs: Vec<Vec<char>>,
    hs: Vec<Vec<char>>,
}

fn read_input() -> Input {
    input! {
        t: usize,
        n: usize,
        vs: [Chars; n],
        hs: [Chars; n - 1],
        board2: [[i64; n]; n]
    }

    let mut board = DynamicMap2d::new_with(0, n);
    let mut legal_actions = DynamicMap2d::new_with(vec![], n);
    let mut cost = 0;

    for i in 0..n {
        for j in 0..n {
            let coord = Coord::new(i, j);
            let coord_down = Coord::new(i + 1, j);
            let coord_right = Coord::new(i, j + 1);

            board[coord] = board2[i][j];

            if i + 1 < n && hs[i][j] == '0' {
                legal_actions[coord].push((DIRS_MAP[&'D'], coord_down));
                legal_actions[coord_down].push((DIRS_MAP[&'U'], coord));
                cost += (board2[i][j] - board2[i + 1][j]).pow(2);
            }
            if j + 1 < n && vs[i][j] == '0' {
                legal_actions[coord].push((DIRS_MAP[&'R'], coord_right));
                legal_actions[coord_right].push((DIRS_MAP[&'L'], coord));
                cost += (board2[i][j] - board2[i][j + 1]).pow(2);
            }
        }
    }

    Input {
        t,
        n,
        n2: n * n,
        cost,
        board,
        legal_actions,
        vs,
        hs,
    }
}

#[derive(Debug, Clone)]
struct TimeKeeper {
    start_time: std::time::Instant,
    time_threshold: f64,
}

impl TimeKeeper {
    fn new(time_threshold: f64) -> Self {
        TimeKeeper {
            start_time: std::time::Instant::now(),
            time_threshold,
        }
    }
    #[inline]
    fn isTimeOver(&self) -> bool {
        let elapsed_time = self.start_time.elapsed().as_nanos() as f64 * 1e-9;
        #[cfg(feature = "local")]
        {
            elapsed_time * 0.55 >= self.time_threshold
        }
        #[cfg(not(feature = "local"))]
        {
            elapsed_time >= self.time_threshold
        }
    }
    #[inline]
    fn get_time(&self) -> f64 {
        let elapsed_time = self.start_time.elapsed().as_nanos() as f64 * 1e-9;
        #[cfg(feature = "local")]
        {
            elapsed_time * 0.55
        }
        #[cfg(not(feature = "local"))]
        {
            elapsed_time
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct Coord {
    row: usize,
    col: usize,
}

impl Coord {
    pub fn new(row: usize, col: usize) -> Self {
        Self { row, col }
    }
    pub fn in_map(&self, height: usize, width: usize) -> bool {
        self.row < height && self.col < width
    }
    pub fn to_index(&self, width: usize) -> CoordIndex {
        CoordIndex(self.row * width + self.col)
    }
}

impl std::ops::Add<CoordDiff> for Coord {
    type Output = Coord;
    fn add(self, rhs: CoordDiff) -> Self::Output {
        Coord::new(
            self.row.wrapping_add_signed(rhs.dr),
            self.col.wrapping_add_signed(rhs.dc),
        )
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CoordDiff {
    dr: isize,
    dc: isize,
}

impl CoordDiff {
    pub const fn new(dr: isize, dc: isize) -> Self {
        Self { dr, dc }
    }
}

pub const ADJ: [CoordDiff; 4] = [
    CoordDiff::new(1, 0),
    CoordDiff::new(!0, 0),
    CoordDiff::new(0, 1),
    CoordDiff::new(0, !0),
];

pub struct CoordIndex(pub usize);

impl CoordIndex {
    pub fn new(index: usize) -> Self {
        Self(index)
    }
    pub fn to_coord(&self, width: usize) -> Coord {
        Coord {
            row: self.0 / width,
            col: self.0 % width,
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct DynamicMap2d<T> {
    pub size: usize,
    map: Vec<T>,
}

impl<T> DynamicMap2d<T> {
    pub fn new(map: Vec<T>, size: usize) -> Self {
        assert_eq!(size * size, map.len());
        Self { size, map }
    }
}

impl<T: Clone> DynamicMap2d<T> {
    pub fn new_with(v: T, size: usize) -> Self {
        let map = vec![v; size * size];
        Self::new(map, size)
    }
    pub fn to_2d_vec(&self) -> Vec<Vec<T>> {
        let mut ret = vec![vec![]; self.size];
        for i in 0..self.map.len() {
            let row = i / self.size;
            ret[row].push(self.map[i].clone());
        }
        ret
    }
}

impl<T> std::ops::Index<Coord> for DynamicMap2d<T> {
    type Output = T;

    #[inline]
    fn index(&self, coordinate: Coord) -> &Self::Output {
        &self[coordinate.to_index(self.size)]
    }
}

impl<T> std::ops::IndexMut<Coord> for DynamicMap2d<T> {
    #[inline]
    fn index_mut(&mut self, coordinate: Coord) -> &mut Self::Output {
        let size = self.size;
        &mut self[coordinate.to_index(size)]
    }
}

impl<T> std::ops::Index<CoordIndex> for DynamicMap2d<T> {
    type Output = T;

    fn index(&self, index: CoordIndex) -> &Self::Output {
        unsafe { self.map.get_unchecked(index.0) }
    }
}

impl<T> std::ops::IndexMut<CoordIndex> for DynamicMap2d<T> {
    #[inline]
    fn index_mut(&mut self, index: CoordIndex) -> &mut Self::Output {
        unsafe { self.map.get_unchecked_mut(index.0) }
    }
}

#[cfg(feature = "local")]
mod visualizer {
    use svg::node::element::path::Data;
    use svg::node::element::{Circle, Group, Line, Path, Rectangle, Style, Text, Title};
    use svg::node::Text as TextContent;
    use svg::Document;

    const MARGIN: f32 = 10.0;

    pub fn doc(height: f32, width: f32) -> Document {
        Document::new()
            .set(
                "viewBox",
                (
                    -MARGIN,
                    -MARGIN,
                    width + 2.0 * MARGIN,
                    height + 2.0 * MARGIN,
                ),
            )
            .set("width", width + MARGIN)
            .set("height", height + MARGIN)
            .set("style", "background-color:#F2F3F5")
    }

    pub fn rect(x: f32, y: f32, w: f32, h: f32, fill: &str) -> Rectangle {
        Rectangle::new()
            .set("x", x)
            .set("y", y)
            .set("width", w)
            .set("height", h)
            .set("fill", fill)
    }

    pub fn cir(x: usize, y: usize, r: usize, fill: &str) -> Circle {
        Circle::new()
            .set("cx", x)
            .set("cy", y)
            .set("r", r)
            .set("fill", fill)
    }

    pub fn lin(x1: usize, y1: usize, x2: usize, y2: usize, color: &str) -> Line {
        Line::new()
            .set("x1", x1)
            .set("y1", y1)
            .set("x2", x2)
            .set("y2", y2)
            .set("stroke", color)
            .set("stroke-width", 3)
            .set("stroke-linecap", "round")
            .set("stroke-linecap", "round")
            .set("marker-end", "url(#arrowhead)")
        // .set("stroke-dasharray", 5)
    }

    pub fn arrow(doc: Document) -> Document {
        doc.add(TextContent::new(
        r#"<defs>
            <marker id="arrowhead" markerWidth="10" markerHeight="7" refX="3" refY="2" orient="auto">
                <polygon points="0 0, 4 2, 0 4" fill="lightgray"/>
            </marker>
        </defs>"#,
    ))
    }

    pub fn txt(x: f32, y: f32, text: &str) -> Text {
        Text::new()
            .add(TextContent::new(text))
            .set("x", x)
            .set("y", y)
            .set("fill", "black")
    }

    // 0 <= val <= 1
    pub fn color(mut val: f64) -> String {
        val = val.min(1.0);
        val = val.max(0.0);
        let (r, g, b) = if val < 0.5 {
            let x = val * 2.0;
            (
                30. * (1.0 - x) + 144. * x,
                144. * (1.0 - x) + 255. * x,
                255. * (1.0 - x) + 30. * x,
            )
        } else {
            let x = val * 2.0 - 1.0;
            (
                144. * (1.0 - x) + 255. * x,
                255. * (1.0 - x) + 30. * x,
                30. * (1.0 - x) + 70. * x,
            )
        };
        format!(
            "#{:02x}{:02x}{:02x}",
            r.round() as i32,
            g.round() as i32,
            b.round() as i32
        )
    }

    pub fn group(title: String) -> Group {
        Group::new().add(Title::new().add(TextContent::new(title)))
    }

    pub fn partition(mut doc: Document, h: &[Vec<char>], v: &[Vec<char>], size: f32) -> Document {
        let H = v.len();
        let W = h[0].len();
        for i in 0..H + 1 {
            for j in 0..W {
                // Entrance
                // if i == 0 && j == ENTRANCE {
                //     continue;
                // }
                if (i == 0 || i == H) || h[i - 1][j] == '1' {
                    let data = Data::new()
                        .move_to((size * j as f32, size * i as f32))
                        .line_by((size * 1.0, 0));
                    let p = Path::new()
                        .set("d", data)
                        .set("stroke", "black")
                        .set("stroke-width", 3.0)
                        .set("stroke-linecap", "round");
                    doc = doc.add(p);
                }
            }
        }
        for j in 0..W + 1 {
            for i in 0..H {
                // Entrance
                // if j == 0 && i == ENTRANCE {
                //     continue;
                // }
                if (j == 0 || j == W) || v[i][j - 1] == '1' {
                    let data = Data::new()
                        .move_to((size * j as f32, size * i as f32))
                        .line_by((0, size * 1.0));
                    let p = Path::new()
                        .set("d", data)
                        .set("stroke", "black")
                        .set("stroke-width", 3.0)
                        .set("stroke-linecap", "round");
                    doc = doc.add(p);
                }
            }
        }
        doc
    }

    pub fn vis(N: usize, vs: &[Vec<char>], hs: &[Vec<char>], board: &[Vec<i64>]) {
        let height = 800.0;
        let width = 800.0;
        let N2 = N * N;
        let d = height / N as f32;
        let mut doc = doc(height, width);
        doc = doc.add(Style::new(format!(
            "text {{text-anchor: middle; dominant-baseline: central; font-size: {}}}",
            10
        )));

        for i in 0..N {
            for j in 0..N {
                let rec = rect(
                    j as f32 * d,
                    i as f32 * d,
                    d,
                    d,
                    &color(board[i][j] as f64 / N2 as f64),
                );
                let text = txt(
                    j as f32 * d + d / 2.0,
                    i as f32 * d + d / 2.0,
                    &board[i][j].to_string(),
                );
                let mut grp = group(format!("(i, j) = ({}, {})\n{}", i, j, board[i][j]));
                grp = grp.add(rec);
                if N <= 20 {
                    grp = grp.add(text);
                }
                doc = doc.add(grp);
            }
        }
        doc = partition(doc, hs, vs, d);
        let vis = format!("<html><body>{}</body></html>", doc);
        std::fs::write("vis.html", vis).unwrap();
    }
}
