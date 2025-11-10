from src.causalchange.util.old.utils_SI import *



# The optimal Sliced Mutual Information
def optimal_SI (X, Y, num_slices, omega_X=math.pi/4, omega_Y=math.pi/4, max_iter=2, lam=.6, lamMI=1., device="cpu"):

    f1 = Net(X[0].shape[0],Y[0].shape[0],X[0].shape[0]).to(device)
    f1_op = optim.Adam(f1.parameters(), lr=0.005, betas=(0.5, 0.999))
    f2 = Net(X[0].shape[0],Y[0].shape[0],Y[0].shape[0]).to(device)
    f2_op= optim.Adam(f2.parameters(), lr=0.005, betas=(0.5, 0.999))


    if isinstance(X, np.ndarray):
        X = torch.tensor(X.astype(np.float32))
    if isinstance(Y, np.ndarray):
        Y = torch.tensor(Y.astype(np.float32))
    embedding_dim_X = X[0].shape[0]
    embedding_dim_Y = Y[0].shape[0]

    for _ in range(max_iter):

        MI=0
        H=0
        
        for _ in range(num_slices):
            pro_X=rand_slices(embedding_dim_X, num_slices=1).to(device)
            pro_Y = rand_slices(embedding_dim_Y, num_slices=1).to(device)


            theta=f1.forward([torch.Tensor(pro_X),torch.Tensor(pro_Y)])
            phi= f2.forward([torch.Tensor(pro_X),torch.Tensor(pro_Y)])

            X_theta= torch.mm(theta,X.t())
            Y_phi= torch.mm(phi,Y.t())


            H+=entropy(X_theta.t())

            MI+=I_est(X_theta.t(),Y_phi.t())/num_slices


            

        PSI = rand_slices(embedding_dim_X, num_slices).to(device)
        UPSILON = rand_slices(embedding_dim_Y, num_slices).to(device)

        THETA= f1.forward([torch.Tensor(PSI),torch.Tensor(UPSILON)])
        arccosX = arccos_distance_torch(THETA,THETA)

        PHI= f2.forward([torch.Tensor(PSI),torch.Tensor(UPSILON)])
        arccosY = arccos_distance_torch(PHI,PHI)
        

        reg=-lam * (arccosX-omega_X) -lam * (arccosY-omega_Y)
        loss = reg- lamMI*MI
        f1_op.zero_grad()
        f2_op.zero_grad()
        loss.backward(retain_graph=True)
        f1_op.step()
        f2_op.step()
    
    


    MI=0
    H=0
    for _ in range(num_slices):
        pro_X=rand_slices(embedding_dim_X, num_slices=1).to(device)
        pro_Y = rand_slices(embedding_dim_Y, num_slices=1).to(device)


        theta=f1.forward([torch.Tensor(pro_X),torch.Tensor(pro_Y)])
        phi= f2.forward([torch.Tensor(pro_X),torch.Tensor(pro_Y)])

        X_theta= torch.mm(theta,X.t())
        Y_phi= torch.mm(phi,Y.t())

        H+=entropy(X_theta.t())/num_slices

        MI+=I_est(X_theta.t(),Y_phi.t())/num_slices
    

    return MI/H
