import numpy as np
import pdb
from pprint import pprint
import pandas as pd


class RandomSCMGenerator():
    '''
    '''
    def __init__(self,num_nodes,max_strength,num_parents,args):
        '''
        max_strength : the strenght of the parents
        num_parents  : the number of parent for each node
        '''
        self.num_nodes = num_nodes
        self.max_strength = max_strength
        self.num_parents = num_parents
        self.args = args #This will make rest of attribute redundant to keep

    def generate_random_adj_mat(self,tol=None):
        '''
        This function will return random graph controlling the following parameters:
        1. strength of connections
        2. sparsity of connection
        '''
        #So right now the tolerange strategy will be 10% of the max strength
        if tol==None:
            tol =self.max_strength/2
         

        #Generating the edges with a particular strength
        A = np.array([np.random.choice([
                                np.random.uniform(-self.max_strength,-tol),
                                np.random.uniform(tol,self.max_strength)
                        ])
                        for _ in range(self.num_nodes*self.num_nodes)]).reshape(
                                                    self.num_nodes,self.num_nodes)
        A = np.tril(A,-1)

        #Next we will ensure number of parents is within max limits
        if self.args["graph_sparsity_method"]=="num_parents":
            for nidx in range(self.num_nodes):
                mask = np.array([1,]*self.num_parents + [0,]*(nidx-self.num_parents))
                np.random.shuffle(mask)
                mask_vec = np.zeros(self.num_nodes)
                mask_vec[0:mask.shape[0]]=mask
                A[nidx,:]=A[nidx,:]*mask_vec
        elif self.args["graph_sparsity_method"]=="adj_dense_prop":
            #First of all we need to generate the mask to based on the prop
            prob_keep = self.args["adj_dense_prop"]
            mask = np.random.choice([0,1],
                                    size=(self.num_nodes,self.num_nodes),
                                    replace=True,
                                    p=[1-prob_keep,prob_keep]
            )
            A = np.multiply(A,mask)
        else:
            raise NotImplementedError
        
        assert np.sum(np.diag(A))==0.0,"Diagnoal non zero"
        
        return A
    
    def generate_gaussian_scm(self,scm_args):
        '''
        '''
        #First of all sample a random graph
        adj_mat = self.generate_random_adj_mat()
        #Generating a random SCM
        scm_args["adj_mat"] = adj_mat
        gSCM = GaussianSCM(scm_args)

        return gSCM


class GaussianSCM:
    '''
    '''
    def __init__(self,args):
        '''
        '''
        self.debug=args["debug_mode"] if "debug_mode" in args else False
        if self.debug:
            print("==================================")
            print("Generating SCM")
        #Paramters for true underlying SCM
        self.noise_mu = np.array(args["noise_mean_list"],dtype=np.float32)
        self.noise_D = np.diag(args["noise_var_list"])
        self.noise_D_list= np.diag(self.noise_D)
        self.dim = self.noise_D.shape[0]
        self.post_do_var=1e-9
        #This adjacency matrix is lower traingular
        self.A = np.array(args["adj_mat"])
        assert np.allclose(self.A, np.tril(self.A)),"adj not lower triangluar"
        #Creating the B matrix from this A
        self.B = np.linalg.inv(np.eye(self.dim)-self.A)
        #TODO: Later we can allow for the random permutation and keep the P for mapping
         
    def _generate_sample(self,num_samples,Ai,noise_Di,noise_mui):
        '''
        Here we will do ancestral sampling using the diagnoal adjacency matrix
        Should we do it ourselves in brute force manner? Use some library or 
        atleast parallelize later
        '''
        #Generating the independent noise for each var and each sample
        standard_noise = np.random.randn(num_samples,self.dim)
        #TODO: Here we have to use sigma (so take sqrt once we have corrected the Di to have variance)
        #For sigma=1 it was same but wont be same for general (done)
        noise = standard_noise*np.sqrt(np.diag(noise_Di)) + noise_mui
        #Now we are ready to generate all the samples
        Bi = np.linalg.inv(np.eye(self.dim)-Ai)
        X = np.matmul(Bi,noise.T).T #to keep the sample x dim shape

        #Generating the covariance matrix to compare later
        #noiseDi is the correct covaraince matrix of noise with varaince in diagonal no sigma
        Si = np.matmul(np.matmul(Bi,noise_Di),Bi.T)
        x_mui = np.matmul(Bi,noise_mui)
        return X,Si,x_mui
    
    def _generate_sample_gamma_noise(self,num_samples,Ai,
                                        noise_Di,obs_noise_gamma_shape):
        '''
        '''
        scale_param = 0.71
        #First of all sampling the noise
        noise = np.random.gamma(obs_noise_gamma_shape,
                                    scale=scale_param,size=(num_samples,self.dim))
        #Zero centring the noise
        noise = noise - obs_noise_gamma_shape*scale_param
        #Now we will apply the filter of internvetion using the proxy of noise D
        Bi = np.linalg.inv(np.eye(self.dim)-Ai)
        X = np.matmul(np.matmul(Bi,noise_Di),noise.T).T 

        #Getting the corresponding mean and variance of X now
        x_mui = np.mean(X,axis=0)
        Si = np.cov(X,rowvar=False)

        # pdb.set_trace()

        return X,Si,x_mui
    
    def _generate_sample_with_atomic_intervention(self,num_samples,intv_args):
        '''
        interv_args: 
            inode : intervened node
            soft_vec : the vector that will signify the soft internvetion
        '''
        if intv_args==None or intv_args["intv_type"]=="obs":
            Ai = self.A.copy()
            noise_Di = self.noise_D.copy()
            noise_mui = self.noise_mu.copy()
        elif intv_args["intv_type"]=="do":
            Ai = self.A.copy()
            Ai[intv_args["inode"],:]=0
            #Updating the noise variance for this node
            noise_Di = self.noise_D.copy()
            noise_Di[intv_args["inode"],intv_args["inode"]]=self.post_do_var
            #Updating the mean of this node too
            noise_mui = self.noise_mu.copy()
            noise_mui[intv_args["inode"]]=intv_args["new_mui"] #the constant it sets to
        elif intv_args["intv_type"]=="hard":
            Ai = self.A.copy()
            Ai[intv_args["inode"],:]=0
            #Updating the noise variance for this node
            noise_Di = self.noise_D.copy()
            noise_Di[intv_args["inode"],intv_args["inode"]]=intv_args["new_vari"]
            #Updating the mean of this node too
            noise_mui = self.noise_mu.copy()
            noise_mui[intv_args["inode"]]=intv_args["new_mui"]
        elif intv_args["intv_type"]=="soft":
            raise NotImplementedError()
            #Otherwise we will perform a soft internvetion
        else:
            raise NotImplementedError()

        #Now finally generating the sample
        if intv_args["noise_type"]=="gaussian":
            X,Si,x_mui = self._generate_sample(num_samples=num_samples,
                                    Ai=Ai,
                                    noise_Di=noise_Di,
                                    noise_mui=noise_mui)
        elif intv_args["noise_type"]=="gamma":
            assert intv_args["intv_type"]=="do" or intv_args["intv_type"]=="obs",\
                            "Other internvetion type not supported"
            noise_Di = np.ones(self.noise_D.shape)
            if intv_args["intv_type"]=="do":
                noise_Di[intv_args["inode"],intv_args["inode"]]=self.post_do_var
            X,Si,x_mui = self._generate_sample_gamma_noise(num_samples=num_samples,
                            Ai=Ai,
                            noise_Di=noise_Di,
                            obs_noise_gamma_shape=intv_args["obs_noise_gamma_shape"],
            )
        else:
            raise NotImplementedError()
        
        #Also generating the covariance for the safe keeping
        true_params=dict(
                        Si = Si,
                        mui = x_mui,
                        Ai = Ai,
        )
        if self.debug:
            print("==================================")
            pprint("Generating samples for:")
            pprint(intv_args)
            pprint("True params:")
            pprint(true_params)

        return X,true_params
    
    def generate_gaussian_mixture(self,intv_type,intv_targets,new_noise_mean,
                                    new_noise_var,num_samples,
                                    noise_type,obs_noise_gamma_shape
        ):
        '''
        '''
        #Genetaing the samples from each of the mixture
        intv_args_dict={}
        mixture_samples=[]
        
        #We will keep the net number of sample same so that the number of component doesnt have any effect
        num_samples = num_samples//(len(intv_targets)+1)
        
        for nidx in intv_targets:
            #Creating the interv args
            intv_args=dict(
                        intv_type=intv_type,
                        inode=nidx,
                        new_mui=new_noise_mean, #need not keep it different
                        new_vari=new_noise_var,
                        noise_type=noise_type,
                        obs_noise_gamma_shape=obs_noise_gamma_shape,
            )
            #Generating the samples for this internvetions
            X,true_params = self._generate_sample_with_atomic_intervention(
                                            num_samples=num_samples,
                                            intv_args=intv_args
            )
            mixture_samples.append(X)
            #Upting the true param of this dist to compare later
            intv_args["true_params"]=true_params
            intv_args_dict[str(nidx)]=intv_args
            intv_args_dict[str(nidx)]["samples"]=X.copy()
        #Adding the observational distribution
        intv_args=dict(intv_type="obs",
                        noise_type=noise_type,
                        obs_noise_gamma_shape=obs_noise_gamma_shape,
        )
        Xobs,obs_true_params = self._generate_sample_with_atomic_intervention(
                                            num_samples=num_samples,
                                            intv_args=intv_args
        )
        mixture_samples.append(Xobs)
        intv_args["true_params"]=obs_true_params
        intv_args_dict["obs"]=intv_args
        intv_args_dict["obs"]["samples"]=Xobs.copy()



        #Consolidating all the samples into one big pile
        mixture_samples = np.concatenate(mixture_samples,axis=0)

        return intv_args_dict,mixture_samples


def generate_mixture_sachs(fpath,num_samples):
    '''
    '''
    #Reading the full dataset
    df = pd.read_csv(fpath,delimiter="\t")


    #Genetaing the samples from each of the mixture
    intv_args_dict={}
    mixture_samples=[]
    intv_targets = [("Akt",3),("PKC",4),("PIP2",5),("Mek",6),("PIP3",7)]
    #Getting the names of the variables
    var_names = df.drop(columns=["experiment"]).columns.tolist()
    var2idx_dict = {var:idx for idx,var in enumerate(var_names)}
    idx2var_dict = {val:key for key,val in var2idx_dict.items()}
    #Getting the adjacecny matrix for this dataset
    A = get_sachs_adj_matrix(var2idx_dict)
    num_nodes = A.shape[0]
    
    #We will keep the net number of sample same so that the number of component doesnt have any effect
    # num_samples = num_samples//(len(intv_targets)+1)

    #Adding the observational data
    print("Getting the observational data")
    intv_args_dict["obs"] = {}
    intv_args_dict["obs"]["tgt_idx"]=None
    obs_samples = df[(df["experiment"]==1) | (df["experiment"]==2)
                        ].drop(columns=["experiment"]).to_numpy()
    print("num samples: obs: ",obs_samples.shape[0])
    mixture_samples.append(obs_samples)
    intv_args_dict["obs"]["samples"]=obs_samples
    intv_args_dict["obs"]["true_params"]=dict(
                        Si = np.cov(obs_samples,rowvar=False),
                        mui = np.mean(obs_samples,axis=0),
                        Ai = A,
    )


    #Now one by one we will add the internvetional data that we know of
    for tgt,expt_num in intv_targets:
        print("Getting the internvetional data: ",tgt)
        #Getting the internvetional data for this target
        intv_samples = df[df["experiment"]==expt_num].drop(
                            columns=["experiment"]).to_numpy()
        print("num_samples: {}: {}".format(tgt,intv_samples.shape[0]))
        mixture_samples.append(intv_samples)
        #Addig the internvetion info
        intv_args_dict[tgt]={}
        intv_args_dict[tgt]["tgt_idx"]=var2idx_dict[tgt]
        #This will have clean mixture samples
        intv_args_dict[tgt]["samples"]=intv_samples

        #Getting the new adjancecy matrix for this intervened dist (do intv)
        Ai = A.copy()
        Ai[var2idx_dict[tgt],:]=0.0
        intv_args_dict[tgt]["true_params"]=dict(
                        Si = np.cov(intv_samples,rowvar=False),
                        mui = np.mean(intv_samples,axis=0),
                        Ai = Ai,
    )
    
    #Acculmulating the samples in to one big matrix
    mixture_samples = np.concatenate(mixture_samples,axis=0)
    print("Total number of samples: ",mixture_samples.shape[0])

    return intv_args_dict,mixture_samples,num_nodes

def get_sachs_adj_matrix(var2idx_dict):
    '''
    '''
    num_nodes = len(var2idx_dict)
    A = np.zeros((num_nodes,num_nodes))
    #Now adding the edges
    A[var2idx_dict["Akt"],var2idx_dict["PKA"]]=1.0
    A[var2idx_dict["Erk"],var2idx_dict["PKA"]]=1.0
    A[var2idx_dict["Mek"],var2idx_dict["PKA"]]=1.0
    A[var2idx_dict["Raf"],var2idx_dict["PKA"]]=1.0
    A[var2idx_dict["JNK"],var2idx_dict["PKA"]]=1.0
    A[var2idx_dict["p38"],var2idx_dict["PKA"]]=1.0
    A[var2idx_dict["Akt"],var2idx_dict["PIP3"]]=1.0
    A[var2idx_dict["PIP2"],var2idx_dict["PIP3"]]=1.0
    A[var2idx_dict["PLCg"],var2idx_dict["PIP3"]]=1.0
    A[var2idx_dict["PKC"],var2idx_dict["PIP2"]]=1.0
    A[var2idx_dict["PIP2"],var2idx_dict["PLCg"]]=1.0
    A[var2idx_dict["PKC"],var2idx_dict["PLCg"]]=1.0
    A[var2idx_dict["Erk"],var2idx_dict["Mek"]]=1.0
    A[var2idx_dict["Mek"],var2idx_dict["Raf"]]=1.0
    A[var2idx_dict["Mek"],var2idx_dict["PKC"]]=1.0
    A[var2idx_dict["Raf"],var2idx_dict["PKC"]]=1.0
    A[var2idx_dict["JNK"],var2idx_dict["PKC"]]=1.0
    A[var2idx_dict["p38"],var2idx_dict["PKC"]]=1.0

    return A 

if __name__=="__main__":
    args={}
    num_nodes=2
    args["noise_mean_list"]=[0.0,0.0]
    args["noise_var_list"]=[1.0,1.0]
    args["adj_mat"]=np.array([
                    [0,0],
                    [1,0]
    ])
    #Creating the SCM 
    gSCM = GaussianSCM(args)
        



    