from __future__ import annotations
from exo import *
from exo import proc
from exo.platforms.rvv import *
from exo.stdlib.scheduling import *

def test_tmp(MR,NR,beta0):
    @proc
    def ukernel_beta0(
            KC: size,
            MR: size,
            NR: size,
            alpha: f32[1],
            A: f32[KC, MR] @ DRAM,
            B: f32[KC, NR] @ DRAM,
            beta: f32[1],
            C: f32[NR, MR] @ DRAM,
            ):   
        for k in seq(0, KC): 
            for j in seq(0, NR): 
                for i in seq(0, MR): 
                    C[j, i] += A[k,i] * B[k,j]
                    
    
    def make_tail(t,desp):
        t = stage_mem(t, "C[_] += _", "C[j,{} + itt]".format(desp), "C_regt", init_zero=beta0)
        t = expand_dim(t, 'C_regt', LANE, 'itt', unsafe_disable_checks=True)
        t = expand_dim(t, 'C_regt', NR, 'j', unsafe_disable_checks=True)
        t = lift_alloc(t, 'C_regt', n_lifts=3)
        #print(t)
        t     = autofission(t, t.find('{}[_] = _'.format('C_regt')).after(), n_lifts=3)
        t = autofission(t, t.find('{}[j,{} + itt] = _'.format('C',desp)).before(), n_lifts=3)
        t = simplify(t)
        t = set_memory(t, 'C_regt', RVV)
        #t = replace_all(t, intrinsics['load'])
        #t = replace_all(t, intrinsics['store'])
        t = simplify(t)
        Buf = 'A'
        Xreg='{}_regt'.format(Buf)
        t = bind_expr(t, '{}[_] #1'.format(Buf),Xreg)
        t = simplify(t)
        t = expand_dim(t, Xreg , LANE, 'itt', unsafe_disable_checks=True)
        t = lift_alloc(t, Xreg, n_lifts=3)
        t = autofission(t, t.find('{}[_] = _'.format(Xreg)).after(),n_lifts=2)
        t = set_memory(t, 'A_regt', RVV)
        #t = replace(t, t.find('for itt in _:_'),intrinsics['load'])
        t = simplify(t)
        scal = 'B'
        scr = '{}_regt'.format(scal)
        t = bind_expr(t, '{}[_] #1'.format(scal),scr)
        #p = stage_mem(p, "for k in _:_", "B[0:KC, 0:{}]".format(NR), "B_reg")
        t = expand_dim(t, scr, LANE, 'itt', unsafe_disable_checks=True)
        t = simplify(t)
        t = expand_dim(t, scr, NR, 'j', unsafe_disable_checks=True)
        t = simplify(t)
        t = lift_alloc(t, scr, n_lifts=3)
        t = autofission(t, t.find('{}[_] = _'.format(scr)).after(), n_lifts=2)
        t = set_memory(t, 'B_regt', RVV)
        #print(t)
        #t = replace(t, t.find('for itt in _:_'),intrinsics['bcast'])
    
        #t = replace(t, t.find('for itt in _:_'),intrinsics['fmla'])

        t = simplify(t)
    
    
        #print(t)
        return t
    
    def reorder_up(p, stmt_pattern, n=1):
        for _ in range(n):
            c = p.find(stmt_pattern).expand(1, 0)
            p = reorder_stmts(p, c)
        return p
    
    def moveup(p,expr):
        while True:
            try:
                p = reorder_up(p, expr)
            except:
                break;
        return p
    
    def unrollbuffers(p,buf):
        p = unroll_buffer(p, buf,0) 
        return p
    
    def make_tail_ok(t):
        t = t.partial_eval(MR=MR, NR=NR)
        #t = set_window(t, "C", True)
        t = stage_mem(t, "C[_] += _", "C[j,i]".format(LANE), "C_reg", init_zero=beta0)
        t = expand_dim(t, 'C_reg', LANE, 'i', unsafe_disable_checks=True)
        t = expand_dim(t, 'C_reg', NR, 'j', unsafe_disable_checks=True)
        t = lift_alloc(t, 'C_reg', n_lifts=3)
        t     = autofission(t, t.find('{}[_] = _'.format('C_reg')).after(), n_lifts=3)
        t = autofission(t, t.find('{}[j,i] = _'.format('C',LANE)).before(), n_lifts=3)
        t = simplify(t)
        t = set_memory(t, 'C_reg', RVV)
        t = replace_all(t, intrinsics['load'] if beta0 == False else  intrinsics['zeros'])
        t = replace_all(t, intrinsics['store'])
        t = simplify(t)
        Buf = 'A'
        Xreg='{}_reg'.format(Buf)
        t = bind_expr(t, '{}[_]'.format(Buf),Xreg)
        t = simplify(t)
        t = expand_dim(t, Xreg , LANE, 'i', unsafe_disable_checks=True)
        t = lift_alloc(t, Xreg, n_lifts=3)
        t = autofission(t, t.find('{}[_] = _'.format(Xreg)).after(),n_lifts=2)
        t = set_memory(t, 'A_reg', RVV)
        t = replace(t, t.find('for i in _:_'),intrinsics['load'])
        t = simplify(t)
        #print(p)
        scal = 'B'
        scr = '{}_reg'.format(scal)
        t = bind_expr(t,scal,scr)
        #p = stage_mem(p, "for k in _:_", "B[0:KC, 0:{}]".format(NR), "B_reg")
        t = expand_dim(t, scr, LANE, 'i', unsafe_disable_checks=True)
        t = simplify(t)
        t = expand_dim(t, scr, NR, 'j', unsafe_disable_checks=True)
        t = simplify(t)
        t = lift_alloc(t, scr, n_lifts=3)
        t = autofission(t, t.find('{}[_] = _'.format(scr)).after(), n_lifts=2)
        t = set_memory(t, 'B_reg', RVV)
        t = replace(t, t.find('for i in _:_'),intrinsics['bcast'])
    
    
        t = replace(t, t.find('for i in _:_'),intrinsics['fmla'])

        t = simplify(t)
    
        while True:
            try:
                t = unroll_loop(t, "j")
            except:
                break;
        t=unrollbuffers(t, "C_reg")
        #for i in range(NR):
        #     t=unrollbuffers(t, "C_reg_{}".format(i))
        #t=unrollbuffers(t, "A_reg")
        t=unrollbuffers(t, "B_reg")
    
        return t

    #MR=5
    #NR=3
    LANE=4
    intrinsics = {'load': rvv_vld_4xf32, 'store': rvv_vst_4xf32, 'fmla':  rvv_vfmacc_4xf32_4xf32,
                         'bcast': rvv_broadcast_4xf32, 'zeros': rvv_broadcast_4xf32_0, 'bcast_scalar': rvv_broadcast_4xf32_scalar}
    
    
    p=ukernel_beta0
    p = rename(p, "gemm_{}_{}x{}_b{}_{}_{}".format("RISCV", MR,NR, 0 if beta0 else 1, "col","f32"))
    p = set_window(p, "C", True)
    p = set_window(p, "A", True)
    p = set_window(p, "B", True)
    if MR <= LANE:
       p = make_tail_ok(p)
       p = simplify(p)
       print(p)
       return p
    
    #p = ukernel_beta0
    p = p.partial_eval(MR=MR,NR=NR)
    p = simplify(p)
    loop='i'
    p =  divide_loop(p,loop, LANE, ['{}t'.format(loop),'{}tt'.format(loop)], tail='cut')
    p = simplify(p)
    if MR % LANE != 0:
        p = autofission(p, p.find("for itt in _:_ #1").before(),n_lifts=2)
    p = stage_mem(p, "C[_] += _", "C[j,itt + {} * it]".format(LANE), "C_reg", init_zero=beta0) #REGISTERS FOR MULTIPLE OF LANE
    
    
    # PREPARE REGISTERS FOR MULTIPLE
    p = expand_dim(p, 'C_reg', LANE, 'itt', unsafe_disable_checks=True)
    #if MR != LANE:
    p = expand_dim(p, 'C_reg', MR//LANE, 'it', unsafe_disable_checks=True)
    p = expand_dim(p, 'C_reg', NR, 'j', unsafe_disable_checks=True)
    p = lift_alloc(p, 'C_reg', n_lifts=4)
    p = simplify(p)
    
    #MOVE LOADS AND STORES OF THE CASE WHEN MULTIPLE
    p = autofission(p, p.find('{}[_] = _'.format('C_reg')).after(), n_lifts=4)
    p = autofission(p, p.find('{}[j,itt+{}*it] = _'.format('C',LANE)).before(), n_lifts=4)
    p = simplify(p)
    
    p = set_memory(p, 'C_reg', RVV)
    #p = replace_all(p, intrinsics['load'])
    #p = replace_all(p, intrinsics['store'])
    
    Buf = 'A'
    Xreg='{}_reg'.format(Buf)
    p = bind_expr(p, '{}[_]'.format(Buf),Xreg)
    p = simplify(p)
    loop = 'i'
    p = expand_dim(p, Xreg , LANE, '{}tt'.format(loop), unsafe_disable_checks=True)
    p = expand_dim(p, Xreg, MR//LANE, '{}t'.format(loop), unsafe_disable_checks=True)
    p = lift_alloc(p, Xreg, n_lifts=4)
    p = autofission(p, p.find('{}[_] = _'.format(Xreg)).after(),n_lifts=3)
    p = set_memory(p, 'A_reg', RVV)
    #p = replace(p, p.find('for itt in _:_'),intrinsics['load'])
    
    scal = 'B'
    scr = '{}_reg'.format(scal)
    p = bind_expr(p,scal,scr)
    p = expand_dim(p, scr, LANE, 'itt', unsafe_disable_checks=True)
    p = simplify(p)
    p = expand_dim(p, scr, NR, 'j', unsafe_disable_checks=True)
    p = simplify(p)
    p = lift_alloc(p, scr, n_lifts=4)
    p = autofission(p, p.find('{}[_] = _'.format(scr)).after(), n_lifts=3)
    
    p = simplify(p)
    p = set_memory(p, 'B_reg', RVV)
    #p = replace(p, p.find('for itt in _:_'),intrinsics['bcast'])
    #p = replace(p, p.find('for itt in _:_'),intrinsics['fmla'])
    
    if MR % LANE != 0:
        p = make_tail(p, (MR//LANE)*LANE)
    
    while True:
        try:
            p = unroll_loop(p, "it")
        except:
            break;
    #p = unroll_loop(p,'it')
    #p = unroll_loop(p,'it')
   # p = unroll_loop(p,'it')
   # p = unroll_loop(p,'it')

    
    if MR % LANE != 0:
        p = reorder_up(p, "C_regt : _",n=5)
        p = reorder_up(p, "for j in _:_ #4",n=1)
        p = reorder_up(p, "for j in _:_ #3",n=1)
        p = reorder_up(p, "for j in _:_ #1",n=2)
        #p = fuse(p,'for j in _:_ #0','for j in _:_ #1')
    #p = reorder_stmts(p, "for j in _:_ #3\nfor j in _:_ #4")
    
    
        p = moveup(p, "B_regt : _")
        p = moveup(p, "B_reg : _")
        p = moveup(p, "A_regt : _")
        p = moveup(p, "A_reg : _")
    
        p = reorder_up(p, "for k in _:_ #1",n=1)
        p = fuse(p,'for k in _:_ #0','for k in _:_ #1')
        p = reorder_up(p, "for j in _:_ #4",n=2)
        p = reorder_up(p, "for j in _:_ #3",n=1)
        """
        if MR // LANE == 2:
            up1 = 13 #3 C + 1 Ct + 3 A + 2 B + 3 C -> 13x5 MR//LANE + 1 + MR//LANE + 2 + MR//LANE
            up2 = 9 # 3 C + 1 Ct + 3 A + 2B -> MR//LANE + 1 + MR//LANE + 2
            up3 = 8 # MR//LANE + 1 + MR//LANE + 1
        elif MR // LANE == 1:
            up1 = 9  #10x7 2 + 1 + 2 + 2 + 2
            up2 = 7  #     2 + 1 + 2 + 2
            up3 = 6  #     2 + 1 + 2 + 1
        else:
            up1 = 6  # 5x7 1 + 1 + 1 + 2 + 1
            up2 = 5
            up3 = 4
        """
        up1 = MR//LANE + 1 + MR//LANE + 2 + MR//LANE
        up2 = MR//LANE + 1 + MR//LANE + 2 
        up3 = MR//LANE + 1 + MR//LANE + 1 
        p = reorder_up(p, "for itt in _:_ #{}".format(up1),n=1)
        p = reorder_up(p, "for itt in _:_ #{}".format(up2),n=1)
        p = reorder_up(p, "for itt in _:_ #{}".format(up3),n=1)
        #print("-------------2------------\n",p)
        #a = input()
        #p = fuse(p,'for j in _:_ #4','for j in _:_ #5')
        #print("-------------3------------\n",p)
        #a = input()
        #p = reuse_buffer(p, 'B_reg','B_regt')
        #print("-------------3------------\n",p)
        #a = input()
    else:
        
        p = moveup(p, "B_regt : _")
        p = moveup(p, "B_reg : _")
        p = moveup(p, "A_regt : _")
        p = moveup(p, "A_reg : _")
        #p = fuse(p,'for j in _:_ #6','for j in _:_ #7')
    
    while True:
        try:
            p = unroll_loop(p, "j")
        except:
            break;
    p = replace_all(p,intrinsics['zeros'])
    p = replace_all(p,intrinsics['bcast'])
    p = replace_all(p,intrinsics['store'])
    p = replace_all(p,intrinsics['load'])
    p = replace_all(p,intrinsics['fmla'])
    p = simplify(p)
    if MR % LANE != 0:
        p = reuse_buffer(p, 'B_reg','B_regt')
    p=unrollbuffers(p, "C_reg")
    for i in range(NR):
        p=unrollbuffers(p, "C_reg_{}".format(i))
    p=unrollbuffers(p, "A_reg")
    p=unrollbuffers(p, "B_reg")
    if MR % LANE != 0:
        p=unrollbuffers(p, "C_regt")
    #if MR % LANE != 0:
   #     p=unrollbuffers(p, "B_regt")
    #    p = reuse_buffer(p, 'B_reg','B_regt')
    print("FINAL",p)
    return p
    #p = reorder_stmts(p, "for j in _:_ #3\nfor j in _:_ #4")
    #p = reorder_stmts(p, "for k in _:_ #0\nfor j in _:_ #2")
    #print("pre3",simplify(p))
    #p = simplify(p)
    #p = fuse(p,'for j in _:_ #0','for j in _:_ #1')
    #p = simplify(p)
    #print("pre-post",simplify(p))
    #print(p.find("for j in _:_ #2"))
    #print(p.find("for k in _:_ #1"))
    #p = reorder_stmts(p, "for j in _:_ #3\nfor k in _:_ #1")
    #print("post",simplify(p))

    #if MR % LANE != 0:
    #    scal = 'B'
    #    scr = '{}_regt'.format(scal)
    #    p = bind_expr(p,scal,scr)
    #    p = expand_dim(p, scr, LANE, 'itt', unsafe_disable_checks=True)
    #    p = expand_dim(p, scr, NR, 'j', unsafe_disable_checks=True)
    #    print(p)
    #    p = autofission(p, p.find('{}[_] = _'.format(scr)).after(), n_lifts=2)
    #    p = lift_alloc(p, scr, n_lifts=3)

for i in range(1,25,1):
    for j in range(1,25,1):
        if i == 0 or j == 0:
            continue
        else:
            print("GENERATING {}x{}".format(i,j))
            locals()['uk_{0}x{1}_b{2}'.format(i,j,False)] = test_tmp(MR=i, NR=j, beta0=False)
            locals()['uk_{0}x{1}_b{2}'.format(i,j,True)] = test_tmp(MR=i, NR=j, beta0=True)
