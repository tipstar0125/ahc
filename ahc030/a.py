# https://qiita.com/aplysia/items/c3f2111110ac5043710a

import sys
from collections import deque
from copy import deepcopy
import random
from math import sqrt,erf

random.seed(0)
is_local=True # ローカルテスト時True
def eprint(s):print(s,file=sys.stderr)

# 入力
N,M,eps=list(map(float,input().split()))
N=int(N)
M=int(M)

class Coord:
    def __init__(self,y,x):
        self.y=y
        self.x=x

class Mino:
    def __init__(self,v):
        self.height=0
        self.width=0
        self.coords=[]
        for i in range(0,len(v),2):
            y=v[i]
            x=v[i+1]
            self.height=max(self.height,y+1)
            self.width=max(self.width,x+1)
            self.coords.append(Coord(y,x))

minos=[]
total=0
for _ in range(M):
    v=list(map(int,input().split()))
    minos.append(Mino(v[1:]))
    total+=v[0]

# ローカルテスト用入力(ローカル時コメントアウトを外す)
ps=[]
for _ in range(M):
    y,x=list(map(int,input().split()))
    ps.append(Coord(y,x))
ans=[list(map(int,input().split())) for _ in range(N)]
es=[float(input()) for _ in range(2*N*N)]

# 各ポリオミノを配置できる始点を求める
fixed_board=[[-1 for _ in range(N)] for _ in range(N)]  # 掘削して盤面を絞るときに使用（今は使用していない）(TODO)
candidate_mino_coords=[]
candidate_board_num=1
for mino in minos:
    cands=[]
    for y in range(N-mino.height+1):
        for x in range(N-mino.width+1):
            ok=True
            for coord in mino.coords:
                ny=y+coord.y
                nx=x+coord.x
                # 掘削済みで油田がなければ、置くことはできない(TODO)
                if fixed_board[ny][nx]==0:
                    ok=False
                    break
            if ok:cands.append(Coord(y,x))
    candidate_board_num*=len(cands)
    candidate_mino_coords.append(cands)

# 盤面のパターンが多い場合は一旦諦める
# 1マス掘削でパターン数を減らす(TODO)
if candidate_board_num>int(1e6):exit(0)

# BFSで盤面を生成
candidate_boards=[]
Q=deque()
Q.append((0,[[0 for _ in range(N)] for _ in range(N)]))
while Q:
    cnt,B=Q.popleft()
    if cnt==M:
        ok=True
        for y in range(N):
            for x in range(N):
                # 生成した盤面の油田数が既に掘削した油田数と一致していなければ、候補から外す(TODO)
                if fixed_board[y][x]!=-1 and fixed_board[y][x]!=B[y][x]:
                    ok=False
                    break
            if not ok:break
        if ok:candidate_boards.append(B)
        continue
    
    for st in candidate_mino_coords[cnt]:
        nB=deepcopy(B)
        ok=True
        for mino in minos[cnt].coords:
            ny=st.y+mino.y
            nx=st.x+mino.x
            nB[ny][nx]+=1
            # 生成途中の盤面の油田数が既に掘削した油田数より多くなたら候補から外す(TODO)
            if fixed_board[y][x]!=-1 and fixed_board[y][x]<B[y][x]:
                ok=False
                break
            if not ok:break
        if ok:Q.append((cnt+1,nB))

# クエリパート
turn=0
cost=0.0

def query(coords):
    global turn
    global cost
    k=len(coords)
    turn+=1
    cost+=1/sqrt(k)
    
    def make_query_return():
        if not is_local:return int(input())
        if k==1:
            return ans[coords[0].y][coords[0].x]
        else:
            vs=0.0
            for coord in coords:
                vs+=ans[coord.y][coord.x]
            mean=(k-vs)*eps+vs*(1.0-eps)
            std=sqrt(k*eps*(1.0-eps))
            ret=mean+std*es[turn]
            ret=int(round(ret))
            return max(0,ret)

    q=["q",k]
    for coord in coords:
        q.append(coord.y)
        q.append(coord.x)
    print(*q,flush=True)
    
    ret=make_query_return()
    return ret

# アンサーパート
def answer(coords):
    k=len(coords)
    
    def make_answer_return():
        if not is_local:return int(input())
        ok=True
        cnt=0
        for coord in coords:
            c=ans[coord.y][coord.x]
            if c==0:
                ok=False
                break
            else:cnt+=c
        ok&=cnt==total
        if ok:return 1
        else:return 0

    a=["a",k]
    for coord in coords:
        a.append(coord.y)
        a.append(coord.x)
    print(*a,flush=True)
    
    ret=make_answer_return()
    return ret

# 尤度計算
# https://bowwowforeach.hatenablog.com/entry/2023/08/24/205427?_gl=1*1rve9o5*_gcl_au*MTE5MTM3NjYzLjE2OTEzMjM4ODU
def likelihood(k,cnt,ret):
    mean=(k-cnt)*eps+cnt*(1.0-eps)
    std=sqrt(k*eps*(1.0-eps))
    diff=ret-mean
    
    def prob_in_range(l,r):
        def cdf(x):return 0.5*(1.0+erf(x)/(std*sqrt(2)))
        return cdf(r)-cdf(l)
    
    if ret==0:return prob_in_range(-1e10,diff+0.5)
    else:return prob_in_range(diff-0.5,diff+0.5)

# 規格化
def normalize(prob):
    s=sum(prob)
    for i in range(len(prob)):
        prob[i]/=s
    return prob

# ベイズ推定パート
board_num=len(candidate_boards)
prob=[1/board_num for _ in range(board_num)] # 同確率で初期化

while turn<2*N*N:
    k=30 # 工夫できそう。情報量最大化とか(TODO)
    coords=[]
    st=set()
    while len(st)<k:
        y=random.randrange(N)
        x=random.randrange(N)
        if (y,x) in st:continue
        st.add((y,x))
        coords.append(Coord(y,x))
    ret=query(coords)
    
    for i in range(board_num):
        cnt=0
        for coord in coords:
            cnt+=candidate_boards[i][coord.y][coord.x]
        # 確率がめちゃくちゃ小さくなって、まともに計算できない場合がありそう
        # 規格化で0割りもありえそうなので、対数を取った方がよさそう(TODO)
        prob[i]*=likelihood(k,cnt,ret)
    prob=normalize(prob)

    mx=max(prob)
    idx=prob.index(mx)
    
    if mx>0.8: # 工夫できそう。第二候補との差とか(TODO)
        a=[]
        for y in range(N):
            for x in range(N):
                if candidate_boards[idx][y][x]>0:a.append(Coord(y,x))
        ret=answer(a)
        if ret==1:
            eprint(f"Turn: {turn}")
            eprint(f"Cost: {cost}")
            exit(0)
        else:
            cost+=1.0
            prob[idx]=0.0
