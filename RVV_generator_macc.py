from __future__ import annotations
from exo import *
from exo import proc
from exo.platforms.rvv import *
from exo.stdlib.scheduling import *

@instr("{dst_data} = __riscv_vrgather_vx_f32m1({src_data}, {imm}, {vl});")
def rvv_gather_4xf32(dst: [f32][4] @ RVV, src: [f32][4] @ RVV, imm: index, vl: size):
        assert stride(dst, 0) == 1
        assert stride(src, 0) == 1
        assert imm >= 0
        assert imm < 4
        assert vl >= 0
        assert vl <= 4

        for i in seq(0, vl):
            dst[i] = src[imm]
@instr("{dst_data} = __riscv_vrgather_vx_f16m1({src_data}, {imm}, {vl});")
def rvv_gather_8xf16(dst: [f16][8] @ RVV, src: [f16][8] @ RVV, imm: index, vl: size):
        assert stride(dst, 0) == 1
        assert stride(src, 0) == 1
        assert imm >= 0
        assert imm < 8
        assert vl >= 0
        assert vl <= 8

        for i in seq(0, vl):
            dst[i] = src[imm]

def ukr_rvv(MR,NR,prec,LANE,beta0,swapAB=False, loadB=0, unroll=1):
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
        assert stride(A,1) == 1 
        assert stride(B,1) == 1
        assert stride(C,1) == 1
        
        for k in seq(0, KC): 
            for j in seq(0, NR): 
                for i in seq(0, MR): 
                    C[j, i] += A[k,i] * B[k,j]
                    
    
    def make_tail(t,desp,loadB=0):
        if loadB==1:
            t = stage_mem(t, "C[_] += _", "C[jtt + {} * jt ,{} + itt]".format(LANE, desp), "C_regt", init_zero=beta0) #
        else:
            t = stage_mem(t, "C[_] += _", "C[j,{} + itt]".format(desp), "C_regt", init_zero=beta0)

        t = expand_dim(t, 'C_regt', LANE, 'itt') #, unsafe_disable_checks=True)
        if loadB==1:
            t = expand_dim(t, 'C_regt', NR, 'jtt + {}* jt'.format(LANE)) #, unsafe_disable_checks=True)
        else:
            t = expand_dim(t, 'C_regt', NR, 'j') #, unsafe_disable_checks=True)
        t = lift_alloc(t, 'C_regt', n_lifts=3 if loadB != 1 else 4)
        t     = autofission(t, t.find('{}[_] = _'.format('C_regt')).after(), n_lifts=3 if loadB != 1 else 4)
        if loadB==1:
            t = autofission(t, t.find('{}[jtt + {} * jt,{} + itt] = _'.format('C',LANE,desp)).before(), n_lifts=4)
        else:
            t = autofission(t, t.find('{}[j,{} + itt] = _'.format('C',desp)).before(), n_lifts=3)
        t = simplify(t)
        t = set_memory(t, 'C_regt', RVV)
        t = simplify(t)
        
        Buf = 'A'
        Xreg='{}_regt'.format(Buf)
        t = bind_expr(t, '{}[_] #1'.format(Buf),Xreg)
        t = simplify(t)
        t = expand_dim(t, Xreg , LANE, 'itt') #, unsafe_disable_checks=True)
        t = lift_alloc(t, Xreg, n_lifts=3 if loadB != 1 else 4)
        t = autofission(t, t.find('{}[_] = _'.format(Xreg)).after(),n_lifts=2 if loadB !=1 else 3)
        t = set_memory(t, 'A_regt', RVV)
        t = set_precision(t,'A_regt',precision)
        t = simplify(t)

        if loadB == 0:
            scal = 'B'
            scr = '{}_regt'.format(scal)
            t = bind_expr(t, '{}[_] #1'.format(scal),scr)
            t = expand_dim(t, scr, LANE, 'itt') #, unsafe_disable_checks=True)
            t = simplify(t)
            t = expand_dim(t, scr, NR, 'j') #, unsafe_disable_checks=True)
            t = simplify(t)
            t = lift_alloc(t, scr, n_lifts=3)
            t = autofission(t, t.find('{}[_] = _'.format(scr)).after(), n_lifts=2)
            t = set_memory(t, 'B_regt', RVV)
            t = set_precision(t,'B_regt',precision)
        elif loadB==1:
            scal = 'B'
            scr = '{}_regt'.format(scal)
            scrtmp = '{}_tmpt'.format(scal)
            t = bind_expr(t,scal,scrtmp)
            t = bind_expr(t,scrtmp,scr)
            t = expand_dim(t, scrtmp, LANE, 'jtt')#, unsafe_disable_checks=True)
            t = expand_dim(t, scr, LANE, 'itt')#, unsafe_disable_checks=True)
            t = expand_dim(t, scr, NR, 'jt * {} + jtt'.format(LANE))#, unsafe_disable_checks=True)
            t = lift_alloc(t, scrtmp, n_lifts=4)
            t = lift_alloc(t, scr, n_lifts=4)
            t = simplify(t)
            t = autofission(t, t.find('{}[_] = _'.format(scr)).after(), n_lifts=3)
            t = autofission(t, t.find('{}[_] = _'.format(scrtmp)).after(), n_lifts=2)
            t = replace(t,t.find('for jtt in _:_ #4'),intrinsics['load'])
            t = simplify(t)
            t = replace(t,t.find('for itt in _:_ #6'), intrinsics['gather'])
            t = simplify(t)
            t = unroll_loop(t,'jtt #4') 
            t = simplify(t)
            t = set_memory(t, 'B_regt', RVV)
            t = set_precision(t, "B_regt", precision)
            t = set_memory(t, 'B_tmpt', RVV)
            t = set_precision(t, "B_tmpt", precision)

        else: #macc
            pass; 
            #t = replace_all(t, intrinsics['macc'])

        t = simplify(t)
    
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
    
    def make_tail_ok(t, loadB=0, unroll=1):
        t = t.partial_eval(MR=MR, NR=NR)
        # 1) if we apply gather, we split the j loop
        # then, in case NR % LANE != 0, we dicide the C computing into two areas:
        # 1) for the part where NR is multiple and 2) for tailing
        if NR < LANE and loadB == 1:
            loadB=0
        if loadB == 1:
            t = divide_loop(t,'j', LANE, ['jt','jtt'], tail='cut') #
            t = simplify(t)
            t = stage_mem(t, "C[_] += _", "C[jtt + {} * jt,i]".format(LANE), "C_reg", init_zero=beta0)
            if NR % LANE != 0:
                # This is the registers that will manage the tail
                t = stage_mem(t, "C[_] += _", "C[{} + jtt,i]".format((NR//LANE)*LANE), "C_reg2", init_zero=beta0)
        else:
            t = stage_mem(t, "C[_] += _", "C[j,i]".format(LANE), "C_reg", init_zero=beta0)
        
        
        t = expand_dim(t, 'C_reg', LANE, 'i')#, unsafe_disable_checks=True)
        
        if loadB==1:
            # 2) Here, we need to resize the C_regs and we need to add the size for the tail
            if NR % LANE !=0:
                t = expand_dim(t, 'C_reg', (NR//LANE)*LANE, 'jt * {} + jtt'.format(LANE))#, unsafe_disable_checks=True)
                t = expand_dim(t, 'C_reg2', LANE, 'i')#, unsafe_disable_checks=True)
                t = expand_dim(t, 'C_reg2', NR%LANE, 'jtt'.format(LANE))#, unsafe_disable_checks=True)
            else:
                t = expand_dim(t, 'C_reg', NR, 'jt * {} + jtt'.format(LANE))#, unsafe_disable_checks=True)
        else:
            t = expand_dim(t, 'C_reg', NR, 'j')#, unsafe_disable_checks=True)
        
        # 3) We lift the declarations outside the loop
        t = lift_alloc(t, 'C_reg', n_lifts=3 if loadB != 1 else 4)
        if loadB == 1 and NR % LANE != 0:
            t = lift_alloc(t, 'C_reg2', n_lifts=3)

        t     = autofission(t, t.find('{}[_] = _'.format('C_reg')).after(), n_lifts=3 if loadB != 1 else 4)
        
        # 4) This is more tricky. First we separate the tail from k loop
        # Then we move out the C_reg2 declaration
        # Finally we move the C_reg2 initialization to the top
        if loadB == 1 and NR % LANE != 0:
            t = autofission(t, t.find('for jtt in _:_ #2').before(), n_lifts=1)
            t = autofission(t, t.find('{}[_] = _'.format('C_reg2')).after(), n_lifts=3)
            #if beta0:
            t = reorder_up(t, "for jtt in _:_ #2",n=1)
        # 5) Now, we move the store of C
        if loadB == 1:
                t = autofission(t, t.find('{}[jtt + {} * jt, i] = _'.format('C',LANE)).before(), n_lifts=3 if loadB != 1 else 4)
                if NR % LANE != 0:
        
        # 6) If we are in a tail case we need to move k loop and then fuse it with the normal case one
                    t = autofission(t, t.find('{}[{} + jtt, i] = _'.format('C',(NR//LANE)*LANE)).before(), n_lifts=3)
                    #t = reorder_up(t, "for jtt in _:_ #3",n=1)

                    t = reorder_up(t, "for k in _:_ #1",n=1)
                    t = fuse(t, "for k in _:_ #0","for k in _:_ #1")

        else:
            t = autofission(t, t.find('{}[j,i] = _'.format('C')).before(), n_lifts=3 if loadB != 1 else 4)
        t = simplify(t)
        t = set_memory(t, 'C_reg', RVV)
        if loadB == 1 and NR % LANE != 0:
            t = set_memory(t, 'C_reg2', RVV)
        t = replace_all(t, intrinsics['load'] if beta0 == False else  intrinsics['zeros'])
        t = replace_all(t, intrinsics['store'])
        t = simplify(t)
        Buf = 'A'
        Xreg='{}_reg'.format(Buf)
        t = bind_expr(t, '{}[_]'.format(Buf),Xreg)
        t = simplify(t)
        t = expand_dim(t, Xreg , LANE, 'i')#, unsafe_disable_checks=True)
        t = lift_alloc(t, Xreg, n_lifts=3 if loadB != 1 else 4)
        t = autofission(t, t.find('{}[_] = _'.format(Xreg)).after(),n_lifts=2 if loadB != 1 else 3)
        t = set_memory(t, 'A_reg', RVV)
        t = set_precision(t,'A_reg',precision)
        if loadB == 1 and NR % LANE != 0:
            Buf = 'A'
            Xreg='{}_reg2'.format(Buf)
            t = bind_expr(t, '{}[k,i] #1'.format(Buf),Xreg)
            t = simplify(t)
            t = expand_dim(t, Xreg , LANE, 'i')#, unsafe_disable_checks=True)
            t = lift_alloc(t, Xreg, n_lifts=3)
            t = autofission(t, t.find('{}[_] = _'.format(Xreg)).after(),n_lifts=2)
            t = set_memory(t, 'A_reg2', RVV)
            t = set_precision(t,'A_reg2',precision)
            t = reorder_up(t, "for i in _:_ #2",n=1)
        
        t = simplify(t)

        #bcast
        if loadB == 0:
            scal = 'B'
            scr = '{}_reg'.format(scal)
            t = bind_expr(t,scal,scr)
            t = expand_dim(t, scr, LANE, 'i')#, unsafe_disable_checks=True)
            t = simplify(t)
            t = expand_dim(t, scr, NR, 'j')#, unsafe_disable_checks=True)
            t = simplify(t)
            t = lift_alloc(t, scr, n_lifts=3)
            t = autofission(t, t.find('{}[_] = _'.format(scr)).after(), n_lifts=2)
            t = set_memory(t, 'B_reg', RVV)
            t = set_precision(t,'B_reg',precision)
        #gather
        elif loadB == 1:
            scal = 'B'
            scr = '{}_reg'.format(scal)
            scrtmp = '{}_tmp'.format(scal)
            t = bind_expr(t,scal,scrtmp)
            t = bind_expr(t,scrtmp,scr)
            t = expand_dim(t, scrtmp, LANE, 'jtt')#, unsafe_disable_checks=True)
            t = expand_dim(t, scr, LANE, 'i')#, unsafe_disable_checks=True)
            if NR % LANE !=0:
                t = bind_expr(t, 'B[k,{} + jtt]'.format((NR//LANE)*LANE),'B_tmp2')
                t = bind_expr(t,'B_tmp2','B_reg2')
                t = expand_dim(t, 'B_tmp2', LANE, 'jtt'.format(LANE))#, unsafe_disable_checks=True)
                t = expand_dim(t, 'B_reg2', LANE, 'i'.format(LANE))#, unsafe_disable_checks=True)
                t = expand_dim(t, 'B_reg2', NR%LANE, 'jtt'.format(LANE))#, unsafe_disable_checks=True)
                t = expand_dim(t, scr, (NR//LANE)*LANE, 'jt * {} + jtt'.format(LANE))#, unsafe_disable_checks=True)
            else:
                t = expand_dim(t, scr, NR, 'jt * {} + jtt'.format(LANE))#, unsafe_disable_checks=True)
            t = lift_alloc(t, scrtmp, n_lifts=4)
            t = lift_alloc(t, scr, n_lifts=4)
            if NR % LANE != 0:
                t = lift_alloc(t, 'B_reg2', n_lifts=3)
                t = lift_alloc(t, 'B_tmp2', n_lifts=3)
            t = simplify(t)
            t = autofission(t, t.find('{}[_] = _'.format(scr)).after(), n_lifts=3)
            t = autofission(t, t.find('{}[_] = _'.format(scrtmp)).after(), n_lifts=2)
            if NR % LANE != 0:
                t = autofission(t, t.find('B_reg2[_] = _'.format(scr)).after(), n_lifts=2)
                t = autofission(t, t.find('B_tmp2[_] = _'.format(scrtmp)).after(), n_lifts=2)
                t = reorder_up(t, "for jtt in _:_ #5",n=1)
                t = reorder_up(t, "for jtt in _:_ #6",n=1)
            
            
            if NR % LANE != 0:
                #We first need to swap loops
                if swapAB:
                    t = reorder_up(t, "for jt in _:_ #{}".format(1),n=2)
                    t = reorder_up(t, "for jtt in _:_ #{}".format(4),n=2)
                    t = reorder_up(t, "for jtt in _:_ #{}".format(5),n=2)

                t = replace(t,t.find('for jtt in _:_ #2'),intrinsics['load'])
                t = replace(t,t.find('for jtt in _:_ #3'),intrinsics['load'])
            else:
                t = replace(t,t.find('for jtt in _:_ #1'),intrinsics['load'])
            t = simplify(t)
            
            if NR % LANE != 0:
                t = replace(t,t.find('for i in _:_ #{}'.format(0 if swapAB else 2)), intrinsics['gather'])
                t = replace(t,t.find('for i in _:_ #{}'.format(0 if swapAB else 2)), intrinsics['gather'])
            else:
                t = replace(t,t.find('for i in _:_ #1'), intrinsics['gather'])
            t = simplify(t)
            # UNROLL GATHERS
            if NR % LANE != 0:
                t = unroll_loop(t,'jtt #2') 
                t = unroll_loop(t,'jtt #2') 
            else:
                t = unroll_loop(t,'jtt #1') 
            t = simplify(t)
            t = set_memory(t, 'B_reg', RVV)
            t = set_precision(t, "B_reg", precision)
            t = set_memory(t, 'B_tmp', RVV)
            t = set_precision(t, "B_tmp", precision)
            if NR % LANE != 0:
                t = set_memory(t, 'B_reg2', RVV)
                t = set_precision(t, "B_reg2", precision)
                t = set_memory(t, 'B_tmp2', RVV)
                t = set_precision(t, "B_tmp2", precision)

        # END IF GATHER FOR B
        else: #loadB==2 -> macc
            #no need for intrinsics for B, just use the macc conversion
            pass

        if swapAB == True:
            jj = 'j' if loadB == 0 else 'jt'
            if loadB == 1 and NR % LANE != 0:
                pass
            else:
                t = reorder_up(t, "for {} in _:_ #{}".format(jj,1),n=1)
        t = replace_all(t, intrinsics['load'])
        t = replace_all(t, intrinsics['bcast'])
    
        if loadB != 2:
            t = replace_all(t, intrinsics['fmla'])
        else:
            t = replace_all(t, intrinsics['macc'])

        t = simplify(t)
        if loadB == 1:

            while True:
                try:
                    t = unroll_loop(t, "jtt")
                except:
                    break;
            
            while True:
                try:
                    t = unroll_loop(t, "jt")
                except:
                    break;
        else:
            while True:
                try:
                    t = unroll_loop(t, "j")
                except:
                    break;
        t = simplify(t)
        
        if loadB == 1 and NR % LANE !=0:
            t = reuse_buffer(t, 'A_reg','A_reg2')
            t = reuse_buffer(t, 'B_tmp','B_tmp2')
            t=unrollbuffers(t, "C_reg2")
            t=unrollbuffers(t, "B_reg2")

        t=unrollbuffers(t, "C_reg")
        if loadB != 2:
            t=unrollbuffers(t, "B_reg")
        if unroll > 1:
            t =  divide_loop(t,'k', unroll, ['kt','ktt'], tail='cut')
            t = unroll_loop(t, "ktt")
    
        return t
    
    ##### START
    if prec=="fp32":
        precision="f32"
        LANE=lane
        if LANE == 4:
            intrinsics = {'load': rvv_vld_4xf32, 'store': rvv_vst_4xf32, 'fmla':  rvv_vfmacc_4xf32_4xf32,
                         'bcast': rvv_broadcast_4xf32, 'zeros': rvv_broadcast_4xf32_0, 'bcast_scalar': rvv_broadcast_4xf32_scalar,
                         'gather':rvv_gather_4xf32, 'macc': rvv_vfmacc_4xf32_1xf32}
        else:
            intrinsics = {'load': rvv_vld_8xf32, 'store': rvv_vst_8xf32, 'fmla':  rvv_vfmacc_8xf32_8xf32,
                         'bcast': rvv_broadcast_8xf32, 'zeros': rvv_broadcast_8xf32_0, 'bcast_scalar': rvv_broadcast_8xf32_scalar,
                         'gather':rvv_gather_8xf32, 'macc': rvv_vfmacc_8xf32_1xf32}
    elif prec=="fp16":
        precision="f16"
        if LANE==8:
            intrinsics = {'load': rvv_vld_8xf16, 'store': rvv_vst_8xf16, 'fmla':  rvv_vfmacc_8xf16_8xf16,
                         'bcast': rvv_broadcast_8xf16, 'zeros': rvv_broadcast_8xf16_0, 'bcast_scalar': rvv_broadcast_8xf16_scalar,
                         'gather':rvv_gather_8xf16, 'macc': rvv_vfmacc_8xf16_1xf16}
        else:
            intrinsics = {'load': rvv_vld_16xf16, 'store': rvv_vst_16xf16, 'fmla':  rvv_vfmacc_16xf16_16xf16,
                         'bcast': rvv_broadcast_16xf16, 'zeros': rvv_broadcast_16xf16_0, 'bcast_scalar': rvv_broadcast_16xf16_scalar,
                         'gather':rvv_gather_16xf16, 'macc': rvv_vfmacc_16xf16_1xf16}
    else:
        print("{} not supported yet!".format(prec))
    
    p=ukernel_beta0
    p = rename(p, "gemm_{}_{}x{}_b{}_{}_{}".format("RVV", MR,NR, 0 if beta0 else 1, "col",prec))
    p = set_window(p, "C", True)
    p = set_window(p, "A", True)
    p = set_window(p, "B", True)
    p = set_precision(p, "C", precision)
    p = set_precision(p, "A", precision)
    p = set_precision(p, "B", precision)
    
    if MR <= LANE:
       p = make_tail_ok(p,loadB)
       p = simplify(p)
       return p

    if NR < LANE and loadB == 1:
        loadB = 0
    p = p.partial_eval(MR=MR,NR=NR)
    p = simplify(p)
    loop='i'
    p =  divide_loop(p,loop, LANE, ['{}t'.format(loop),'{}tt'.format(loop)], tail='cut')
    if loadB == 1:
        p = divide_loop(p,'j', LANE, ['jt','jtt'], tail='cut') #
    p = simplify(p)
    if MR % LANE != 0:
        if loadB == 1:
            if NR % LANE == 0:
                p = autofission(p, p.find("for itt in _:_ #1").before(),n_lifts=3)
            else:
                p = autofission(p, p.find("for itt in _:_ #1").before(),n_lifts=2)
        else:
            p = autofission(p, p.find("for itt in _:_ #1").before(),n_lifts=2)
    if loadB == 1:
        if NR % LANE == 0:
            p = stage_mem(p, "C[_] += _", "C[jtt + {} * jt ,itt + {} * it]".format(LANE, LANE), "C_reg", init_zero=beta0) #
        else:
            p = stage_mem(p, "C[_] += _", "C[jtt + {} * jt ,itt + {} * it]".format(LANE, LANE), "C_reg", init_zero=beta0) #
            if MR % LANE != 0:
                p = stage_mem(p, "C[_] += _", "C[jtt + {} * jt, {} + itt]".format(LANE, (MR//LANE)*LANE), "C_reg1", init_zero=beta0) #
            p = stage_mem(p, "C[_] += _", "C[{} + jtt,itt + {} * it]".format((NR//LANE)*LANE, LANE), "C_reg2", init_zero=beta0) #
            if MR % LANE != 0:
                p = stage_mem(p, "C[_] += _", "C[{} + jtt, {} + itt]".format((NR//LANE)*LANE, (MR//LANE)*LANE), "C_reg3", init_zero=beta0) #
    else:
        p = stage_mem(p, "C[_] += _", "C[j,itt + {} * it]".format(LANE), "C_reg", init_zero=beta0) #REGISTERS FOR MULTIPLE OF LANE
    
    # PREPARE REGISTERS FOR MULTIPLE
    p = expand_dim(p, 'C_reg', LANE, 'itt') #, unsafe_disable_checks=True)
    p = expand_dim(p, 'C_reg', MR//LANE, 'it') #, unsafe_disable_checks=True)
    
    if loadB == 1:
        if NR % LANE == 0:
            p = expand_dim(p, 'C_reg', NR, 'jt * {} + jtt'.format(LANE)) #, unsafe_disable_checks=True)
        else:
            p = expand_dim(p, 'C_reg', (NR//LANE)*LANE, 'jt * {} + jtt'.format(LANE)) #, unsafe_disable_checks=True)
            if MR % LANE != 0:
                p = expand_dim(p, 'C_reg1', LANE, 'itt') #, unsafe_disable_checks=True)
                p = expand_dim(p, 'C_reg1', (NR//LANE)*LANE, 'jt * {} + jtt'.format(LANE)) #, unsafe_disable_checks=True)

            p = expand_dim(p, 'C_reg2', LANE, 'itt') #, unsafe_disable_checks=True)
            p = expand_dim(p, 'C_reg2', MR//LANE, 'it') #, unsafe_disable_checks=True)
            p = expand_dim(p, 'C_reg2', NR%LANE, 'jtt') #, unsafe_disable_checks=True)
            
            if MR % LANE != 0:
                p = expand_dim(p, 'C_reg3', LANE, 'itt') #, unsafe_disable_checks=True)
                p = expand_dim(p, 'C_reg3', NR%LANE, 'jtt') #, unsafe_disable_checks=True)
    else:
        p = expand_dim(p, 'C_reg', NR, 'j') #, unsafe_disable_checks=True)
    
    if loadB == 1:
        p = lift_alloc(p, 'C_reg', n_lifts=5)
    else:
        p = lift_alloc(p, 'C_reg', n_lifts=4)


    if loadB == 1 and NR % LANE != 0:
        p = lift_alloc(p, 'C_reg2', n_lifts=4)
        if MR % LANE != 0:
            p = lift_alloc(p, 'C_reg1', n_lifts=4)
            p = lift_alloc(p, 'C_reg3', n_lifts=3)
    p = simplify(p)
    
    
    #MOVE C LOADS 
    if loadB == 1:
        p = autofission(p, p.find('{}[_] = _'.format('C_reg')).after(), n_lifts=5)
    else:
        p = autofission(p, p.find('{}[_] = _'.format('C_reg')).after(), n_lifts=4)


    if loadB == 1 and NR % LANE != 0:
        if MR % LANE != 0:
            p = autofission(p, p.find("for jt in _:_ #2").before(),n_lifts=3)
            p = autofission(p, p.find('{}[_] = _'.format('C_reg1')).after(), n_lifts=4)
            #p = reorder_up(p, "for jt in _:_ #2",n=1)
            #p = fuse(p,'for k in _:_ #0','for k in _:_ #1')

        if MR % LANE == 0:
            p = autofission(p, p.find("for jtt in _:_ #2").before(),n_lifts=3)
        else:
            p = autofission(p, p.find("for jtt in _:_ #4").before(),n_lifts=3)
        p = autofission(p, p.find('{}[_] = _'.format('C_reg2')).after(), n_lifts=4)
        if MR % LANE == 0:
            
            p = reorder_up(p, "for jtt in _:_ #2",n=1)
        else:
            pass
            #p = reorder_up(p, "for jtt in _:_ #4",n=1)
            #p = fuse(p,'for k in _:_ #0','for k in _:_ #1')
        
        if MR % LANE != 0:
            p = autofission(p, p.find("for itt in _:_ #6").before(),n_lifts=3)
            p = autofission(p, p.find('{}[_] = _'.format('C_reg3')).after(), n_lifts=4)
            #p = reorder_up(p, "for jtt in _:_ #6",n=1)
            #p = fuse(p,'for k in _:_ #0','for k in _:_ #1')
    
    # MOVE C STORES ####QUITAR FUSIONES
    if loadB == 1:
        p = autofission(p, p.find('{}[jtt + {} * jt,itt+{}*it] = _'.format('C',LANE,LANE)).before(), n_lifts=5)
        if NR % LANE != 0:
            if MR % LANE != 0:
                p = autofission(p, p.find('{}[jtt + {} * jt, {} + itt] = _'.format('C',LANE,(MR//LANE)*LANE)).before(), n_lifts=4)
            p = autofission(p, p.find('{}[{} + jtt,itt+{}*it] = _'.format('C',(NR//LANE)*LANE,LANE)).before(), n_lifts=4)
            if MR % LANE != 0:
                p = autofission(p, p.find('{}[{} + jtt, {} + itt] = _'.format('C',(NR//LANE)*LANE,(MR//LANE)*LANE)).before(), n_lifts=4)
            if MR % LANE == 0:
                p = reorder_up(p, "for k in _:_ #1",n=1)
                p = fuse(p,'for k in _:_ #0','for k in _:_ #1')
    else:
        p = autofission(p, p.find('{}[j,itt+{}*it] = _'.format('C',LANE)).before(), n_lifts=4)
    p = simplify(p)
    
    p = set_precision(p, "C_reg", precision)
    p = set_memory(p, 'C_reg', RVV)
    
    if loadB == 1 and NR % LANE != 0:
        p = set_precision(p, "C_reg2", precision)
        p = set_memory(p, 'C_reg2', RVV)
        if MR % LANE != 0:
            p = set_precision(p, "C_reg1", precision)
            p = set_memory(p, 'C_reg1', RVV)
            p = set_precision(p, "C_reg3", precision)
            p = set_memory(p, 'C_reg3', RVV)

    Buf = 'A'
    Xreg='{}_reg'.format(Buf)
    p = bind_expr(p, '{}[_]'.format(Buf),Xreg)
    p = simplify(p)
    loop = 'i'
    p = expand_dim(p, Xreg , LANE, '{}tt'.format(loop))#, unsafe_disable_checks=True)
    p = expand_dim(p, Xreg, MR//LANE, '{}t'.format(loop))#, unsafe_disable_checks=True)
    p = lift_alloc(p, Xreg, n_lifts=4 if loadB != 1 else 5)
    p = autofission(p, p.find('{}[_] = _'.format(Xreg)).after(),n_lifts=3 if loadB != 1 else 4)
    p = set_memory(p, 'A_reg', RVV)
    p = set_precision(p, "A_reg", precision)
    p = moveup(p, "A_reg: _")
    p = simplify(p)
    
    if loadB == 1 and NR % LANE != 0:
        Buf = 'A'
        loop = 'i'
        if MR % LANE != 0:
            Xreg='{}_reg1'.format(Buf)
            p = bind_expr(p, '{}[_] #1'.format(Buf),Xreg)
            p = simplify(p)
            p = expand_dim(p, Xreg , LANE, '{}tt'.format(loop))#, unsafe_disable_checks=True)
            p = lift_alloc(p, Xreg, n_lifts=4)
            p = autofission(p, p.find('{}[_] = _'.format(Xreg)).after(),n_lifts=3)
            p = set_memory(p, 'A_reg1', RVV)
            p = set_precision(p, "A_reg1", precision)
            p = moveup(p, "A_reg1: _")
        
        Xreg='{}_reg2'.format(Buf)
        if MR % LANE == 0:
            p = bind_expr(p, '{}[_] #1'.format(Buf),Xreg)
        else:
            p = bind_expr(p, '{}[_] #2'.format(Buf),Xreg)
        p = simplify(p)
        p = expand_dim(p, Xreg , LANE, '{}tt'.format(loop))#, unsafe_disable_checks=True)
        p = expand_dim(p, Xreg, MR//LANE, '{}t'.format(loop))#, unsafe_disable_checks=True)
        p = lift_alloc(p, Xreg, n_lifts=4)
        p = autofission(p, p.find('{}[_] = _'.format(Xreg)).after(),n_lifts=3)
        p = set_memory(p, 'A_reg2', RVV)
        p = set_precision(p, "A_reg2", precision)
        p = moveup(p, "A_reg2: _")
        p = simplify(p)
        if MR % LANE == 0:
            p = reorder_up(p, "for it in _:_ #4",n=1)
        
        if MR % LANE != 0:
            Xreg='{}_reg3'.format(Buf)
            p = bind_expr(p, '{}[_] #3'.format(Buf),Xreg)
            p = simplify(p)
            p = expand_dim(p, Xreg , LANE, '{}tt'.format(loop))#, unsafe_disable_checks=True)
            p = lift_alloc(p, Xreg, n_lifts=3)
            p = autofission(p, p.find('{}[_] = _'.format(Xreg)).after(),n_lifts=2)
            p = set_memory(p, 'A_reg3', RVV)
            p = set_precision(p, "A_reg3", precision)
            p = moveup(p, "A_reg3: _")
        
    
    if loadB == 1:
        scal = 'B'
        scr = '{}_reg'.format(scal)
        scrtmp = '{}_tmp'.format(scal)
        p = bind_expr(p,scal,scrtmp)
        p = bind_expr(p,scrtmp,scr)
        p = expand_dim(p, scrtmp, LANE, 'jtt')#, unsafe_disable_checks=True)
        p = expand_dim(p, scr, LANE, 'itt')#, unsafe_disable_checks=True)
        if NR % LANE == 0:
            p = expand_dim(p, scr, NR, 'jt * {} + jtt'.format(LANE))#, unsafe_disable_checks=True)
        else:
            p = expand_dim(p, scr, (NR//LANE)*LANE, 'jt * {} + jtt'.format(LANE))#, unsafe_disable_checks=True)
        p = lift_alloc(p, scrtmp, n_lifts=5)
        p = lift_alloc(p, scr, n_lifts=5)
        p = set_memory(p, 'B_reg', RVV)
        p = set_memory(p, 'B_tmp', RVV)
        p = set_precision(p, "B_reg", precision)
        p = set_precision(p, "B_tmp", precision)
        p = moveup(p, "B_reg: _")
        p = moveup(p, "B_tmp: _")
        p = simplify(p)
        if NR % LANE != 0:
            if MR % LANE != 0:
                scr1 = '{}_reg1'.format(scal)
                scrtmp1 = '{}_tmp1'.format(scal)
                p = bind_expr(p,'B[_] #1',scrtmp1)
                p = bind_expr(p,scrtmp1,scr1)
                p = expand_dim(p, scrtmp1, LANE, 'jtt')#, unsafe_disable_checks=True)
                p = expand_dim(p, scr1, LANE, 'itt')#, unsafe_disable_checks=True)
                p = expand_dim(p, scr1, (NR//LANE)*LANE, 'jt * {} + jtt'.format(LANE))#, unsafe_disable_checks=True)
                p = lift_alloc(p, scrtmp1, n_lifts=4)
                p = lift_alloc(p, scr1, n_lifts=4)
                p = set_memory(p, 'B_reg1', RVV)
                p = set_memory(p, 'B_tmp1', RVV)
                p = set_precision(p, "B_reg1", precision)
                p = set_precision(p, "B_tmp1", precision)
                p = moveup(p, "B_reg1: _")
                p = moveup(p, "B_tmp1: _")

            scr2 = '{}_reg2'.format(scal)
            scrtmp2 = '{}_tmp2'.format(scal)
            if MR % LANE == 0:
                p = bind_expr(p,'B[_] #1',scrtmp2)
            else:
                p = bind_expr(p,'B[_] #2',scrtmp2)
            p = bind_expr(p,scrtmp2,scr2)
            p = expand_dim(p, scrtmp2, LANE, 'jtt')#, unsafe_disable_checks=True)
            p = expand_dim(p, scr2, LANE, 'itt')#, unsafe_disable_checks=True)
            p = expand_dim(p, scr2, NR%LANE, 'jtt')#, unsafe_disable_checks=True)
            p = lift_alloc(p, scrtmp2, n_lifts=4)
            p = lift_alloc(p, scr2, n_lifts=4)
            p = set_memory(p, 'B_reg2', RVV)
            p = set_memory(p, 'B_tmp2', RVV)
            p = set_precision(p, "B_reg2", precision)
            p = set_precision(p, "B_tmp2", precision)
            p = moveup(p, "B_reg2: _")
            p = moveup(p, "B_tmp2: _")
           
            if MR % LANE != 0:
                scr3 = '{}_reg3'.format(scal)
                scrtmp3 = '{}_tmp3'.format(scal)
                p = bind_expr(p,'B[_] #3',scrtmp3)
                p = bind_expr(p,scrtmp3,scr3)
                p = expand_dim(p, scrtmp3, LANE, 'jtt')#, unsafe_disable_checks=True)
                p = expand_dim(p, scr3, LANE, 'itt')#, unsafe_disable_checks=True)
                p = expand_dim(p, scr3, NR%LANE, 'jtt'.format(LANE))#, unsafe_disable_checks=True)
                p = lift_alloc(p, scrtmp3, n_lifts=3)
                p = lift_alloc(p, scr3, n_lifts=3)
                p = set_memory(p, 'B_reg3', RVV)
                p = set_memory(p, 'B_tmp3', RVV)
                p = set_precision(p, "B_reg3", precision)
                p = set_precision(p, "B_tmp3", precision)
                p = moveup(p, "B_reg3: _")
                p = moveup(p, "B_tmp3: _")


        p = autofission(p, p.find('{}[_] = _'.format(scr)).after(), n_lifts=4)
        p = autofission(p, p.find('{}[_] = _'.format(scrtmp)).after(), n_lifts=2)
        if NR % LANE != 0:
            if MR % LANE != 0:
                p = autofission(p, p.find('{}[_] = _'.format(scr1)).after(), n_lifts=3)
                p = autofission(p, p.find('{}[_] = _'.format(scrtmp1)).after(), n_lifts=2)
            
            p = autofission(p, p.find('{}[_] = _'.format(scr2)).after(), n_lifts=3)
            p = autofission(p, p.find('{}[_] = _'.format(scrtmp2)).after(), n_lifts=2)
            
            if MR % LANE != 0:
                p = autofission(p, p.find('{}[_] = _'.format(scr3)).after(), n_lifts=2)
                p = autofission(p, p.find('{}[_] = _'.format(scrtmp3)).after(), n_lifts=2)
            if MR % LANE == 0: 
                p = reorder_up(p, "for jtt in _:_ #5",n=1)
                p = reorder_up(p, "for jtt in _:_ #6",n=1)
                if swapAB:
                    p = reorder_up(p, "for jt in _:_ #1",n=2)
                    p = reorder_up(p, "for jtt in _:_ #4",n=2)
                    p = reorder_up(p, "for jtt in _:_ #5",n=2)
            else: # MR % LANE !=0
                # FIRST MOVE ALL LOOPS!!!
                # C LOADS
                p = reorder_up(p, "for jt in _:_ #4",n=1) #Creg1
                p = reorder_up(p, "for jt in _:_ #3",n=1) #Creg1
                
                p = reorder_up(p, "for jtt in _:_ #10",n=1) #Creg2
                p = reorder_up(p, "for jtt in _:_ #9",n=1) #Creg2
                p = reorder_up(p, "for jtt in _:_ #6",n=1) #Creg2
                p = reorder_up(p, "for jtt in _:_ #5",n=1) #Creg2
                
                p = reorder_up(p, "for jtt in _:_ #15",n=1) #Creg3
                p = reorder_up(p, "for jtt in _:_ #14",n=1) #Creg3
                p = reorder_up(p, "for jtt in _:_ #11",n=1) #Creg3
                p = reorder_up(p, "for jtt in _:_ #10",n=1) #Creg3
                p = reorder_up(p, "for jtt in _:_ #7",n=1) #Creg3
                p = reorder_up(p, "for jtt in _:_ #6",n=1) #Creg3

                # NOW WE MOVE Ks Up (or C store down)
                p = reorder_up(p, "for k in _:_ #1",n=1) #KXreg1
                p = reorder_up(p, "for k in _:_ #2",n=2) #KXreg2
                p = reorder_up(p, "for k in _:_ #3",n=3) #KXreg3

                p = fuse(p,'for k in _:_ #0','for k in _:_ #1')
                p = fuse(p,'for k in _:_ #0','for k in _:_ #1')
                p = fuse(p,'for k in _:_ #0','for k in _:_ #1')
                #MOVE B PREV TO A SO WE CAN SWAP EASYLY
                p = reorder_up(p, "for jt in _:_ #2",n=1) #Breg
                p = reorder_up(p, "for jt in _:_ #4",n=2) #Breg1
                p = reorder_up(p, "for jt in _:_ #3",n=1) #Breg1i
                p = reorder_up(p, "for jt in _:_ #3",n=1) #Breg1 #This one move breg1 over breg because we eant to reuse the breg
                p = reorder_up(p, "for jtt in _:_ #10",n=2) #Breg2
                p = reorder_up(p, "for jtt in _:_ #9",n=2) #Breg2
                p = reorder_up(p, "for jtt in _:_ #8",n=1) #Breg2
                p = reorder_up(p, "for jtt in _:_ #11",n=2) #Breg2
                p = reorder_up(p, "for jtt in _:_ #10",n=2) #Breg2
                p = reorder_up(p, "for jtt in _:_ #9",n=1) #Breg2
                p = reorder_up(p, "for jtt in _:_ #13",n=2) #Breg3
                p = reorder_up(p, "for jtt in _:_ #12",n=2) #Breg3
                p = reorder_up(p, "for jtt in _:_ #11",n=2) #Breg3
                p = reorder_up(p, "for jtt in _:_ #10",n=1) #Breg3
                p = reorder_up(p, "for jtt in _:_ #14",n=2) #Breg3
                p = reorder_up(p, "for jtt in _:_ #13",n=2) #Breg3
                p = reorder_up(p, "for jtt in _:_ #12",n=2) #Breg3
                p = reorder_up(p, "for jtt in _:_ #11",n=1) #Breg3
                
                # NOW MOVE THE As
                # Areg is fine so just move it if no swap
                p = reorder_up(p, "for it in _:_ #2",n=0) #Areg #6
                p = reorder_up(p, "for itt in _:_ #10",n=1) #Areg1
                p = reorder_up(p, "for it in _:_ #4",n=2) #Areg2
                p = reorder_up(p, "for itt in _:_ #14",n=1) #Areg3
                p = reorder_up(p, "for itt in _:_ #13",n=1) #Areg3
                p = reorder_up(p, "for itt in _:_ #12",n=1) #Areg3
                
                if swapAB == False:
                    p = reorder_up(p, "for it in _:_ #2",n=6) #Areg
                    p = reorder_up(p, "for itt in _:_ #9",n=1) #Areg1
                    p = reorder_up(p, "for itt in _:_ #8",n=2) #Areg1
                    p = reorder_up(p, "for itt in _:_ #7",n=2) #Areg1
                    p = reorder_up(p, "for itt in _:_ #6",n=1) #Areg1
                    p = reorder_up(p, "for it in _:_ #3",n=7) #Areg2
                    p = reorder_up(p, "for itt in _:_ #11",n=1) #Areg3
                    p = reorder_up(p, "for itt in _:_ #10",n=2) #Areg3
                    p = reorder_up(p, "for itt in _:_ #9",n=2) #Areg3
                    p = reorder_up(p, "for itt in _:_ #8",n=1) #Areg3
                
                if beta0 == True:
                    p = replace_all(p,intrinsics['zeros'])
                
                p = simplify(p)
                p = replace_all(p,intrinsics['load'])
                p = replace_all(p,intrinsics['store'])
                #LOAD of Bs
                #p = replace(p,p.find('for jtt in _:_ #4'),intrinsics['load'])
               # p = replace(p,p.find('for jtt in _:_ #5'),intrinsics['load'])
                #p = replace(p,p.find('for jtt in _:_ #6'),intrinsics['load'])
                #p = replace(p,p.find('for jtt in _:_ #7'),intrinsics['load'])
                p = replace(p,p.find('for itt in _:_ #0'), intrinsics['gather'])
                p = replace(p,p.find('for itt in _:_ #0'), intrinsics['gather'])
                p = replace(p,p.find('for itt in _:_ #0'), intrinsics['gather'])
                p = replace(p,p.find('for itt in _:_ #0'), intrinsics['gather'])
                
                p = replace(p,p.find('for itt in _:_ #0'), intrinsics['fmla'])
                p = replace(p,p.find('for itt in _:_ #0'), intrinsics['fmla'])
                p = replace(p,p.find('for itt in _:_ #0'), intrinsics['fmla'])
                p = replace(p,p.find('for itt in _:_ #0'), intrinsics['fmla'])
                p = simplify(p)
                while True:
                    try:
                        p = unroll_loop(p, "it")
                    except:
                        break;
                  
                while True:
                    try:
                        p = unroll_loop(p, "jtt")
                    except:
                        break;
                
                while True:
                    try:
                        p = unroll_loop(p, "jt")
                    except:
                        break;
            
                p = reuse_buffer(p, 'A_reg','A_reg2')
                p = reuse_buffer(p, 'A_reg1','A_reg3')
                p = reuse_buffer(p, 'B_reg','B_reg1')
                p = reuse_buffer(p, 'B_reg2','B_reg3')
                p = reuse_buffer(p, 'B_tmp','B_tmp1')
                p = reuse_buffer(p, 'B_tmp','B_tmp2')
                p = reuse_buffer(p, 'B_tmp','B_tmp3')
                p = simplify(p)
                p=unrollbuffers(p, "C_reg")
                p=unrollbuffers(p, "C_reg1")
                p=unrollbuffers(p, "C_reg2")
                p=unrollbuffers(p, "C_reg3")
                for i in range((NR//LANE)*LANE):
                    p=unrollbuffers(p, f"C_reg_{i}")
                    #p=unrollbuffers(p, f"C_reg1_{i}")
                for i in range(NR%LANE):
                    p=unrollbuffers(p, f"C_reg2_{i}")

                p=unrollbuffers(p, "B_reg")
                p=unrollbuffers(p, "B_reg2")
                p=unrollbuffers(p, "A_reg")
                if unroll > 1:
                    p =  divide_loop(p,'k', unroll, ['kt','ktt'], tail='cut')
                    p = unroll_loop(p, "ktt")
                return p
        
        
        
        
        if NR % LANE == 0:
            p = replace(p,p.find('for jtt in _:_ #1'),intrinsics['load'])
            p = simplify(p)
            p = replace(p,p.find('for itt in _:_ #2'), intrinsics['gather'])
        else:
            p = replace(p,p.find('for jtt in _:_ #2'),intrinsics['load'])
            p = replace(p,p.find('for jtt in _:_ #3'),intrinsics['load'])
            p = simplify(p)
            if swapAB:
                p = replace(p,p.find('for itt in _:_ #2'), intrinsics['gather'])
                p = replace(p,p.find('for itt in _:_ #2'), intrinsics['gather'])
            else:
                p = replace(p,p.find('for itt in _:_ #4'), intrinsics['gather'])
                p = replace(p,p.find('for itt in _:_ #4'), intrinsics['gather'])
        p = simplify(p)
        if NR % LANE == 0:
            p = unroll_loop(p,'jtt #1') 
        p = simplify(p)
        p = set_memory(p, 'B_reg', RVV)
        p = set_precision(p, "B_reg", precision)
        p = set_memory(p, 'B_tmp', RVV)
        p = set_precision(p, "B_tmp", precision)
        if NR % LANE != 0:
            p = set_memory(p, 'B_reg2', RVV)
            p = set_precision(p, "B_reg2", precision)
            p = set_memory(p, 'B_tmp2', RVV)
            p = set_precision(p, "B_tmp2", precision)
    elif loadB == 0: #bcast
        scal = 'B'
        scr = '{}_reg'.format(scal)
        p = bind_expr(p,scal,scr)
        p = expand_dim(p, scr, LANE, 'itt')#, unsafe_disable_checks=True)
        p = simplify(p)
        p = expand_dim(p, scr, NR, 'j')#, unsafe_disable_checks=True)
        p = simplify(p)
        p = lift_alloc(p, scr, n_lifts=4)
        p = autofission(p, p.find('{}[_] = _'.format(scr)).after(), n_lifts=3)
        p = simplify(p)
        p = set_memory(p, 'B_reg', RVV)
        p = set_precision(p, "B_reg", precision)
    else: #macc
        pass


    if MR % LANE != 0:
        p = make_tail(p, (MR//LANE)*LANE, loadB)

    while True:
        try:
            p = unroll_loop(p, "it")
        except:
            break;

    if MR % LANE != 0:
        p = reorder_up(p, "C_regt : _",n=5 if loadB !=1 else 6)
        ll = 'j' if loadB != 1 else 'jt'

        if loadB != 2:
            p = reorder_up(p, f"for {ll} in _:_ #4",n=1) #-> Debería ser el 3?
            p = reorder_up(p, f"for {ll} in _:_ #3",n=1)
            p = reorder_up(p, f"for {ll} in _:_ #1",n=2)
        else:
            p = reorder_up(p, f"for {ll} in _:_ #3",n=1) #-> Debería ser el 3?
            p = reorder_up(p, f"for {ll} in _:_ #2",n=1)

        if loadB != 2: 
            p = moveup(p, "B_regt : _")
            p = moveup(p, "B_reg : _")
            if loadB == 1:
                p = moveup(p, "B_tmp : _")
                p = moveup(p, "B_tmpt : _")
        p = moveup(p, "A_regt : _")
        p = moveup(p, "A_reg : _")
    
        p = reorder_up(p, "for k in _:_ #1",n=1)
        p = fuse(p,'for k in _:_ #0','for k in _:_ #1')
        ll = 'j' if loadB != 1 else 'jt'
        if loadB != 2:      
            p = reorder_up(p, f"for {ll} in _:_ #4",n=2)
            p = reorder_up(p, f"for {ll} in _:_ #3",n=1)
 
        if loadB != 2:
            up1 = MR//LANE + 1 + MR//LANE + 2 + MR//LANE
            up2 = MR//LANE + 1 + MR//LANE + 2 
            up3 = MR//LANE + 1 + MR//LANE + 1 
        
            if loadB == 1:
                up1 = up1-2
                up2 = up2-2
                up3 = up2
            p = reorder_up(p, "for itt in _:_ #{}".format(up1),n=1)
            p = reorder_up(p, "for itt in _:_ #{}".format(up2),n=1)
            p = reorder_up(p, "for itt in _:_ #{}".format(up3),n=1)
        else:
            up1 = MR//LANE + 1 + MR//LANE + MR//LANE
            p = reorder_up(p, "for itt in _:_ #{}".format(up1),n=1)

        if swapAB:
            upswap = MR//LANE + 1
            jl = 'j' if loadB == 0 else 'jt'
            p = reorder_up(p, "for {} in _:_ #{}".format(jl,2),n=upswap)
            p = reorder_up(p, "for {} in _:_ #{}".format(jl,3),n=upswap)
        

    else: # MR%LANE == 0
        
        if swapAB:
            upswap = MR//LANE
            if loadB == 1:
                if NR % LANE == 0:
                    p = reorder_up(p, "for jt in _:_ #{}".format(1),n=upswap)
                else:
                    pass
            else:

                p = reorder_up(p, "for j in _:_ #{}".format(1),n=upswap)
        
        if loadB == 1: 
            p = moveup(p,"B_tmp : _")
            if NR % LANE != 0:
                p = moveup(p,"B_tmp2 : _")
                p = moveup(p,"B_reg2 : _")
        if loadB == 1 and NR % LANE !=0:
            p = moveup(p,"A_reg2 : _")
        if loadB != 2:
            p = moveup(p, "B_regt : _")
            p = moveup(p, "B_reg : _")
        p = moveup(p, "A_regt : _")
        p = moveup(p, "A_reg : _")
    # end else
    
    while True:
        try:
            p = unroll_loop(p, "j")
        except:
            break;
    while True:
        try:
            p = unroll_loop(p, "jtt")
        except:
            break;

    while True:
        try:
            p = unroll_loop(p, "jt")
        except:
            break;
    p = simplify(p)

    if loadB == 1:
        while True:
            try:
                p = replace(p,p.find('for itt in _:_ ') ,intrinsics['zeros'])
            except:
                break;
        while True:
            try:
                p = replace(p,p.find('for itt in _:_ ') , intrinsics['load'])
            except:
                break;
        while True:
            try:
                p = replace(p,p.find('for itt in _:_ ') , intrinsics['fmla'])
            except:
                break;
        while True:
            try:
                p = replace(p,p.find('for itt in _:_ ') , intrinsics['store'])
            except:
                break;
    else:

        p = replace_all(p,intrinsics['zeros'])
        p = replace_all(p,intrinsics['bcast'])
        p = replace_all(p,intrinsics['load'])
        if loadB == 2:
            p = replace_all(p,intrinsics['macc'])
        else:
            p = replace_all(p,intrinsics['fmla'])
        p = replace_all(p,intrinsics['store'])
    p = simplify(p)
    
    if MR % LANE != 0:
        if loadB != 2:
            p = reuse_buffer(p, 'B_reg','B_regt')
            if loadB == 1:
                p = reuse_buffer(p, 'B_tmp','B_tmpt')
    
    if loadB == 1 and NR % LANE != 0:
        p = reuse_buffer(p, 'A_reg','A_reg2')
        p=unrollbuffers(p, "B_reg2")
        p = reuse_buffer(p, 'B_tmp','B_tmp2')
    
    p=unrollbuffers(p, "C_reg")
    if loadB == 1 and NR % LANE !=0:
        for i in range((NR//LANE)*LANE):
            p=unrollbuffers(p, "C_reg_{}".format(i))
        p=unrollbuffers(p, "C_reg2".format(i))
        for i in range(NR%LANE):
            p=unrollbuffers(p, "C_reg2_{}".format(i))

    else:
        for i in range(NR):
            p=unrollbuffers(p, "C_reg_{}".format(i))
    
    p=unrollbuffers(p, "A_reg")
    if loadB != 2:
        p=unrollbuffers(p, "B_reg")
    
    if MR % LANE != 0:
        p=unrollbuffers(p, "C_regt")
    if unroll > 1:
        p =  divide_loop(p,'k', unroll, ['kt','ktt'], tail='cut')
        p = unroll_loop(p, "ktt")
    return p


def howmanyregs(M,N, lane, loadB):
    reg_a = M//lane if M % lane == 0 else M//lane + 1
    if loadB == 2:
        reg_b = 0
    else:
        reg_b = N
    reg_c = N *  (M//lane if M % lane == 0 else M//lane + 1)
    return reg_a + reg_b + reg_c


m, n, lane, pre, swapAB, loadB, unroll, regs  = (int(x) for x in input().split())
#lane = 4
pr="fp{}".format(pre)
swapAB = bool(swapAB)
if loadB == 2:
    swapAB=False
## loadB 0=bcast,1=gather,2=fmacc
#gather = bool(gather)
mr = emr = m
nr = enr = n
print(m, n, lane, pre, swapAB, loadB)
maxi = maxj = 0

if howmanyregs(emr, enr, lane, loadB) <= regs:
    for i in range(1,mr+1,1):
        for j in range(1,nr+1,1):
            print("GENERATING {}x{} with {} registers".format(i,j, howmanyregs(i,j,lane, loadB)))
            locals()['uk_{0}x{1}_b{2}'.format(i,j,False)] = ukr_rvv(MR=i, NR=j, prec=pr, LANE = lane, beta0=False, swapAB=swapAB, loadB=loadB, unroll=unroll)
            locals()['uk_{0}x{1}_b{2}'.format(i,j,True)]  = ukr_rvv(MR=i, NR=j, prec=pr, LANE = lane, beta0=True,  swapAB=swapAB, loadB=loadB, unroll=unroll)

#from generate_matrix import generate_file

#generate_file(maxi,maxj,lane,'RVV',pr)
