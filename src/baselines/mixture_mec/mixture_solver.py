import numpy as np
import itertools as it
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import GridSearchCV
from pprint import pprint
import pdb
import pandas as pd
import json
import pathlib
import multiprocessing as mp
import pandas as pd


from pgmpy.estimators import PC
from causaldag import unknown_target_igsp,igsp
import causaldag as cd
from causaldag import partial_correlation_suffstat, partial_correlation_test, MemoizedCI_Tester
from causaldag import gauss_invariance_suffstat, gauss_invariance_test, MemoizedInvarianceTester


from src.baselines.mixture_mec.scm_module import *

class GaussianMixtureSolver():
    '''
    '''
    def __init__(self,dtype):
        '''
        '''
        self.dtype = dtype

    def get_best_estimated_matching_error(self,intv_args_dict,gm,debug=False):
        '''
        '''
        #print("Getting the best maching of estimated params!")
        weight_list = gm.weights_
        mean_list=gm.means_
        cov_list = gm.covariances_

        est_comp_idx_list = list(range(len(mean_list)))
        actual_tgt_list = list(intv_args_dict.keys())

        #If the est comp is more we will make perm of est comp
        est_more = None
        if len(est_comp_idx_list)>len(actual_tgt_list):
            comp_perm_list = it.permutations(est_comp_idx_list,
                                            r=len(actual_tgt_list),
            )
            est_more = True
        else:
            comp_perm_list = it.permutations(actual_tgt_list,
                                            r=len(est_comp_idx_list),
            )
            est_more=False
        
        #Now we will start matching
        min_err = float("inf")
        min_perm=None
        min_weight_precision_error = None
        component_true_weight = 1.0/len(mean_list) #equal proportion
        for req_comp_perm in comp_perm_list:
            err = 0.0
            weight_precision_error = 0.0
            #Now we have a perm: match comp_perm --> actual_tgt_list
            # req_comp_perm = comp_perm[0:len(actual_tgt_list)]
            # assert len(req_comp_perm)==len(actual_tgt_list)

            #Creating the zip object to iterate
            if est_more:
                zip_obj =  zip(req_comp_perm,actual_tgt_list)
            else:
                zip_obj = zip(est_comp_idx_list,req_comp_perm)
            
            for cidx,comp in zip_obj:
                #So we will only add the error of the matched component
                #The error in the rest of the components are thrown away
                mean_err = np.sum(
                    np.abs(mean_list[cidx]-intv_args_dict[comp]["true_params"]["mui"])
                )#/np.sum(np.abs(intv_args_dict[comp]["true_params"]["mui"])+1e-7)
                cov_error = np.sum(
                    np.abs(cov_list[cidx]-intv_args_dict[comp]["true_params"]["Si"])
                )#/np.sum(np.abs(intv_args_dict[comp]["true_params"]["Si"])+1e-7)
                
                #Getting the accumulated error
                weight_precision_error += np.abs(
                    weight_list[cidx]-component_true_weight
                )
                err+=mean_err+cov_error
            #Now checking if this perm/matching gives minimum error
            if err<min_err:
                min_err=err 
                min_perm=req_comp_perm
                min_weight_precision_error=weight_precision_error
        
        '''
        Note: In the min-error estimation we dont add the error for the case
        when we have less number of component whereas the actual tgt are more.
        So if we dont predict anything --> parameter estimation will be zero
        So, param est error alone is not a good metric to judge the result.


        But it is fine that:
        1. we dont asssume that our model has estimated those parameters 
            as zero which could be wrong (cuz pi=0 then the param could be arbit even inf)
        2. Also we report the number of component anyway 
        '''
        
        if debug:
            print("error:",min_err)
            print("min_perm:",min_perm)
        
        #Creating zip object to update
        if est_more:
            merge_zip_obj = zip(min_perm,actual_tgt_list)
        else:
            merge_zip_obj = zip(est_comp_idx_list,min_perm)

        #Updating the interv dict with appropriate perm
        for cidx,comp in merge_zip_obj:
            intv_args_dict[comp]["est_params"]={}
            intv_args_dict[comp]["est_params"]["mui"] = mean_list[cidx]
            intv_args_dict[comp]["est_params"]["Si"] = cov_list[cidx]
        
        #WE need to add more component only if there are more componenet
        if est_more:
            #Actually we should have rest of the component too
            rest_perm = set(est_comp_idx_list).difference(min_perm)
            for ridx in rest_perm:
                #Creating new component
                intv_args_dict["left_comp_"+str(ridx)]={}
                #Adding the true param same as the estimated (for UTGSP oracle)
                intv_args_dict["left_comp_"+str(ridx)]["true_params"]={}
                intv_args_dict["left_comp_"+str(ridx)]["true_params"]["mui"]=mean_list[ridx]
                intv_args_dict["left_comp_"+str(ridx)]["true_params"]["Si"]=cov_list[ridx]
                #Estimated is also same (we are not computing the error so fine to keep same)
                intv_args_dict["left_comp_"+str(ridx)]["est_params"]={}
                intv_args_dict["left_comp_"+str(ridx)]["est_params"]["mui"]=mean_list[ridx]
                intv_args_dict["left_comp_"+str(ridx)]["est_params"]["Si"]=cov_list[ridx]
        #print("Best Matching Done!")

        return min_err,min_perm,intv_args_dict,min_weight_precision_error

    def middle_out_component_selector(self,log_lik_list,cutoff_drop_ratio):
        '''
        '''
        # print("fwd")
        fwd_cutoff_idx = len(log_lik_list)-1
        for idx in range(1,len(log_lik_list)):
            if ((np.abs(log_lik_list[idx]-log_lik_list[idx-1])
                            )/np.abs(log_lik_list[idx-1]))<cutoff_drop_ratio:
                fwd_cutoff_idx=idx-1
                break
        
        # print("back")
        bwd_cutoff_idx = 0 
        for idx in range(len(log_lik_list)-1,0,-1):
            #print("\nbwidx:{},change_ratio:{}",idx,
            #        ((np.abs(log_lik_list[idx]-log_lik_list[idx-1])
            #            )/np.abs(log_lik_list[idx])))
            if ((np.abs(log_lik_list[idx]-log_lik_list[idx-1])
                        )/np.abs(log_lik_list[idx]))>cutoff_drop_ratio:
                bwd_cutoff_idx=idx
                break
        #print("fwd_cutoff_idx: ",fwd_cutoff_idx)
        #print("bwd_cutoff_idx: ",bwd_cutoff_idx)
        
        # num_component = int(((fwd_cutoff_idx+1)+(bwd_cutoff_idx+1))/2)
        num_component = bwd_cutoff_idx+1
        return num_component

    def mixture_disentangler(self,max_component,
                intv_args_dict,
                mixture_samples,
                tol,cutoff_drop_ratio,
                debug=False,
                bic_sel=False):
        #Performing the mixture model selection
        gm_score_dict=None
        if bic_sel:
            param_grid = {
                "n_components": range(1,max_component+1),
                "tol":[tol],
                }
            def gmm_bic_score(estimator, X):
                """Callable to pass to GridSearchCV that will use the BIC score."""
                # Make it negative since GridSearchCV expects a score to maximize
                return -estimator.bic(X)
            
            grid_search = GridSearchCV(
                GaussianMixture(), param_grid=param_grid, scoring=gmm_bic_score
            )
            grid_search.fit(mixture_samples)
            #Getting the best estimator
            gm = grid_search.best_estimator_
            est_component=len(gm.weights_)

            #Getting the score dict 
            df = pd.DataFrame(grid_search.cv_results_)[
                ["param_n_components", "mean_test_score"]
            ]
            df["mean_test_score"] = -df["mean_test_score"]
            gm_score_dict=dict(zip(df.param_n_components,df.mean_test_score))
        else:
            #Here we will perform component selection
            gm_score_dict = {}
            log_lik_list = []
            gm_dict={}
            for num_component in range(1,max_component+1):
                #Now we are ready run the mini disentanglement algos
                gm_comp = GaussianMixture(n_components=num_component,
                                            tol=tol,
                                            random_state=0,

                ).fit(mixture_samples)
                log_lik_list.append(gm_comp.lower_bound_)
                gm_dict[num_component]=gm_comp
                gm_score_dict[num_component]=gm_comp.lower_bound_
            #None number of component not allowed!
            if debug: print(log_lik_list)
            est_component = self.middle_out_component_selector(
                                log_lik_list,
                                cutoff_drop_ratio)
            gm=gm_dict[est_component]
        if debug: print("Number of component selected: ",est_component)

        if debug:
            print("==================================")
            print("Estimated Means (unmatched):")
            pprint(gm.means_*(gm.means_>1e-5))
            print("==================================")
            pprint("Estimated Covarainces (unmatched):")
            pprint(gm.covariances_*(gm.covariances_>1e-5))
        
        min_err,min_perm,min_weight_precision_error = None,None,None
        if self.dtype=="simulation" or self.dtype=="sachs":
            min_err,min_perm,intv_args_dict,min_weight_precision_error\
                                = self.get_best_estimated_matching_error(
                                                intv_args_dict,gm)

        return min_err,intv_args_dict,min_weight_precision_error,est_component,gm_score_dict,gm
    
    def run_pc_over_each_component(self,intv_args_dict,num_samples):
        '''
        '''
        sample_per_comp = num_samples//len(intv_args_dict.keys())
        #Iterating over all the estimated component and 
        for comp in intv_args_dict.keys():
            est_mui = intv_args_dict[comp]["est_params"]["mui"]
            est_Si = intv_args_dict[comp]["est_params"]["Si"]

            #Now we will generate samples from this distribution
            samples = pd.DataFrame(
                        np.random.multivariate_normal(est_mui,
                                                    est_Si,
                                                    size=sample_per_comp),
                        columns=[str(idx) for idx in range(est_mui.shape[0])]
            )

            #Now we will run the CI test
            pc = PC(samples)
            #Check significance level (fisher transform CI test)
            pdag = pc.skeleton_to_pdag(*pc.build_skeleton(ci_test='pearsonr'))
            intv_args_dict[comp]["est_pdag"] = pdag
            # pdb.set_trace()
    
        return intv_args_dict
    
    def identify_intervention_utigsp(self,intv_args_dict,num_samples,run_igsp=False):
        '''
        They assume we have access to the observational data
        code taken from UTGSP tutorial
        '''
        sample_per_comp = num_samples//len(intv_args_dict.keys())
        num_nodes = intv_args_dict["obs"]["true_params"]["mui"].shape[0]

        #Generating the observational samples
        # obs_true_mui = intv_args_dict["obs"]["true_params"]["mui"]
        # obs_true_Si = intv_args_dict["obs"]["true_params"]["Si"]
        # obs_samples = np.random.multivariate_normal(obs_true_mui,
        #                                         obs_true_Si,
        #                                         size=sample_per_comp)
        obs_samples=intv_args_dict["obs"]["samples"][0:sample_per_comp,:]
        

        #Generating the observational samples for oracle
        # if self.dtype=="simulation":
        #     obs_true_mui = intv_args_dict["obs"]["true_params"]["mui"]
        #     obs_true_Si = intv_args_dict["obs"]["true_params"]["Si"]
        #     oracle_obs_samples = np.random.multivariate_normal(obs_true_mui,
        #                                             obs_true_Si,
        #                                             size=sample_per_comp)
        oracle_obs_samples=intv_args_dict["obs"]["samples"][0:sample_per_comp,:]
        

        #Generating the samples from the estimated params
        # obs_est_mui = intv_args_dict["obs"]["est_params"]["mui"]
        # obs_est_Si = intv_args_dict["obs"]["est_params"]["Si"]
        # obs_est_samples = np.random.multivariate_normal(obs_est_mui,
        #                                         obs_est_Si,
        #                                         size=sample_per_comp)
        
        #Generating the sample from a random component other thatn obs
        #This will test that UTGSP will not need the obs distribution
        # intv_locs = set(intv_args_dict.keys())
        # intv_locs.remove("obs")
        # intv_base_loc = np.random.choice(list(intv_locs),size=1)[0]
        # intv_base_est_mui = intv_args_dict[intv_base_loc]["est_params"]["mui"]
        # intv_base_est_Si = intv_args_dict[intv_base_loc]["est_params"]["Si"]
        # intv_est_samples = np.random.multivariate_normal(
        #                                         intv_base_est_mui,
        #                                         intv_base_est_Si,
        #                                         size=sample_per_comp)
        


        #Iterating over all the estimated component but first adding the obs first
        #This is required by the UGSP
        est_actual_target_list = ["obs",]
        oracle_actual_target_list = ["obs",]
        utarget_sample_list  = [obs_samples.copy(),]
        utarget_oracle_sample_list = [oracle_obs_samples.copy(),]
        igsp_setting_list = [dict(interventions=[]),]
        igsp_sample_list = [oracle_obs_samples.copy(),]

        for comp in intv_args_dict.keys():
            #Skipping the obs cuz already added above
            if comp=="obs":
                continue
            if run_igsp and "left_comp" not in comp:
                if self.dtype=="simulation":
                    igsp_setting_list.append(dict(
                                                interventions=[int(comp)]
                    ))
                    igsp_intv_samples = intv_args_dict[comp]["samples"][0:sample_per_comp,:]
                    igsp_sample_list.append(igsp_intv_samples)
                elif self.dtype=="sachs":
                    igsp_setting_list.append(dict(
                                    interventions=[
                                        int(intv_args_dict[comp]["tgt_idx"])]
                    ))
                    #Adding the sample
                    igsp_intv_samples = intv_args_dict[comp]["samples"][0:sample_per_comp,:]
                    igsp_sample_list.append(igsp_intv_samples)
                else:
                    raise NotImplementedError()

            '''
            So here in case we have estimated less num of comp
            we will not have the estimated params for those actual targets
            Thus this if statement.
            '''
            if "est_params" in intv_args_dict[comp]:
                est_actual_target_list.append(comp)
                #Generating the samples from the estimated parameters
                est_mui = intv_args_dict[comp]["est_params"]["mui"]
                est_Si = intv_args_dict[comp]["est_params"]["Si"]
                #Now we will generate samples from this distribution
                intv_samples = np.random.multivariate_normal(est_mui,
                                                        est_Si,
                                                        size=sample_per_comp)
                utarget_sample_list.append(intv_samples)



            #Genearting the samples for the oracle using the exact parameters
            # if "left_comp_" in comp:
            #     oracle_mui = intv_args_dict[comp]["true_params"]["mui"]
            #     oracle_Si = intv_args_dict[comp]["true_params"]["Si"]
            #     oracle_intv_samples = np.random.multivariate_normal(oracle_mui,
            #                                             oracle_Si,
            #                                             size=sample_per_comp)
            #     utarget_oracle_sample_list.append(oracle_intv_samples)
            #     oracle_actual_target_list.append(comp)
            # elif self.dtype=="simulation" or self.dtype=="sachs":
            #Only add to the oracel which are the true targets in the mixture
            if "left_comp_" not in comp:
                oracle_intv_samples = intv_args_dict[comp]["samples"][0:sample_per_comp,:]
                utarget_oracle_sample_list.append(oracle_intv_samples)
                oracle_actual_target_list.append(comp)
            # else:
            #     raise NotImplementedError()
        
        #Creating the suddicient statistics
        obs_suffstat = partial_correlation_suffstat(obs_samples)
        #This is obs suff stat for the oracle
        oracle_obs_suffstat = partial_correlation_suffstat(oracle_obs_samples)
        #This is base suff stat when we dont want to use the obs as base in UTGSP
        # intv_base_suffstat = partial_correlation_suffstat(intv_est_samples)


        #Getting the interventional suff stat
        invariance_suffstat = gauss_invariance_suffstat(obs_samples, 
                                                utarget_sample_list)
        oracle_invariance_suffstat = gauss_invariance_suffstat(oracle_obs_samples, 
                                                utarget_oracle_sample_list)
        #Getting the intv suff sata with intv base instead of obs
        # intv_base_invariance_suffstat = gauss_invariance_suffstat(
        #                                         intv_est_samples, 
        #                                         utarget_sample_list
        # )
        #Getting intv suff stat for the igsp
        igsp_invariance_suffstat = gauss_invariance_suffstat(
                                                obs_samples, 
                                                igsp_sample_list
        )

        #CI tester and invariance tester
        if self.dtype=="simulation":
            alpha = 1e-3
            alpha_inv = 1e-3
        elif self.dtype=="sachs":
            alpha = 1e-3
            alpha_inv = 1e-3
        else:
            raise NotImplementedError()
        ci_tester = MemoizedCI_Tester(partial_correlation_test, 
                                        obs_suffstat, alpha=alpha)
        oracle_ci_tester = MemoizedCI_Tester(partial_correlation_test, 
                                        oracle_obs_suffstat, alpha=alpha)
        # intv_base_ci_tester = MemoizedCI_Tester(partial_correlation_test, 
        #                                 intv_base_suffstat, alpha=alpha)
        
        
        invariance_tester = MemoizedInvarianceTester(
                                        gauss_invariance_test,
                                        invariance_suffstat, 
                                        alpha=alpha_inv)
        oracle_invariance_tester = MemoizedInvarianceTester(
                                        gauss_invariance_test,
                                        oracle_invariance_suffstat, 
                                        alpha=alpha_inv)
        # inv_base_invariance_tester = MemoizedInvarianceTester(
        #                                 gauss_invariance_test,
        #                                 intv_base_invariance_suffstat, 
        #                                 alpha=alpha_inv)
        igsp_invariance_tester = MemoizedInvarianceTester(
                                        gauss_invariance_test,
                                        igsp_invariance_suffstat, 
                                        alpha=alpha_inv)
        
        #Runnng UTGSP
        setting_list = [dict(known_interventions=[]) for _ in range(len(est_actual_target_list))]
        #print("Running UTGSP")
        est_dag, est_targets_list = unknown_target_igsp(setting_list, 
                                                set(list(range(num_nodes))), 
                                                ci_tester, 
                                                invariance_tester)
        
        #Running the Oracle-UTGSP (i.e with sample from correct params)
        setting_list = [dict(known_interventions=[]) for _ in range(len(oracle_actual_target_list))]
        #print("Running Oracle-UTGSP")
        oracle_est_dag, oracle_est_targets_list = unknown_target_igsp(setting_list, 
                                                set(list(range(num_nodes))), 
                                                oracle_ci_tester, 
                                                oracle_invariance_tester)
        
        #Running the UTGSP with different base instead of obs dist
        intv_base_est_dag, intv_base_est_targets_list = None, None
        # setting_list = [dict(known_interventions=[]) for _ in range(len(est_actual_target_list))]
        # print("Running intv_base-UTGSP")
        # intv_base_est_dag, intv_base_est_targets_list = unknown_target_igsp(setting_list, 
        #                                         set(list(range(num_nodes))), 
        #                                         intv_base_ci_tester, 
        #                                         inv_base_invariance_tester)

        #Running the IGSP to see the upper bound of the estimation
        #Question: Should we put extra obs data similar to UTGSP here?
        if run_igsp:
            #print("Running the IGSP")
            igsp_est_dag = igsp(igsp_setting_list,
                                set(list(range(num_nodes))),
                                oracle_ci_tester, 
                                igsp_invariance_tester,
            )
        else:
            igsp_est_dag=None

        #Here we are not matching the target using the similarty but 
        #rather than the parameter which is already done in the step1
        #unlike our last project where we matched target using JS.
        #Adding the estimated targets to the acutal targets dct
        for act_tgt,est_tgt in zip(est_actual_target_list,est_targets_list):
            intv_args_dict[act_tgt]["est_tgt"]=list(est_tgt)
        
        for act_tgt,oracle_est_tgt in zip(oracle_actual_target_list,oracle_est_targets_list):
            intv_args_dict[act_tgt]["oracle_est_tgt"]=list(oracle_est_tgt)

        # for act_tgt,intv_base_est_tgt in zip(est_actual_target_list,intv_base_est_targets_list):
        #     intv_args_dict[act_tgt]["intv_base_est_tgt"]=list(intv_base_est_tgt)
        
        return est_dag,intv_args_dict,oracle_est_dag,igsp_est_dag,intv_base_est_dag


#SINGLE EXPERIMENT KERNEL
def run_mixture_disentangle(args):
    '''
    '''
    #Collecting the metrics to evaluate the results later
    metric_dict={}

    if args["dtype"]=="simulation":
        #Creating the SCM
        gargs={}
        gargs["noise_mean_list"]=[args["obs_noise_mean"],]*args["num_nodes"]
        gargs["noise_var_list"]=[args["obs_noise_var"],]*args["num_nodes"]
        scmGen = RandomSCMGenerator(num_nodes=args["num_nodes"],
                                    max_strength=args["max_edge_strength"],
                                    num_parents=args["num_parents"],
                                    args=args,
        )
        gSCM = scmGen.generate_gaussian_scm(scm_args=gargs)

        
        #Step 0: Generating the samples and interventions configs
        #print("Generating mixture samples!")
        intv_args_dict,mixture_samples = gSCM.generate_gaussian_mixture(
                                                            args["intv_type"],
                                                            args["intv_targets"],
                                                            args["new_noise_mean"],
                                                            args["new_noise_var"],
                                                            args["mix_samples"],
                                                            args["noise_type"],
                                                            args["obs_noise_gamma_shape"],
        )
    elif args["dtype"]=="sachs":
        #Getting the samples and the intervention configs
        #print("Generating the mixture sample")
        intv_args_dict,mixture_samples,num_nodes = generate_mixture_sachs(
                                            args["dataset_path"],
                                            args["mix_samples"],
        )
        args["num_nodes"]=num_nodes
    else:
        raise NotImplementedError()

    #Step 1: Running the disentanglement
    #print("Step 1: Disentangling Mixture")
    gSolver = GaussianMixtureSolver(args["dtype"])
    #We will allow number of component = n+1 (hopefully it will find zero weight)
    err,intv_args_dict,weight_precision_error,est_num_comp,gm_score_dict \
                                    = gSolver.mixture_disentangler(
                                                    args["num_tgt_prior"],
                                                    intv_args_dict,
                                                    mixture_samples,
                                                    args["gmm_tol"],
                                                    args["cutoff_drop_ratio"],
                                                    )
    metric_dict["param_est_rel_err"]=err
    metric_dict["est_num_comp"]=est_num_comp
    metric_dict["weight_precision_error"]=weight_precision_error
    metric_dict["gm_score_dict"]=gm_score_dict
    #print("error:",err)
    #print("weight precision error:",metric_dict["weight_precision_error"])
    





    #Step 2: Finding the graph for each component
    #print("Stage 2: Estimating individual graph using PC")
    # gSolver.run_pc_over_each_component(intv_args_dict,args["stage2_samples"])
    est_dag,intv_args_dict,oracle_est_dag,igsp_est_dag,intv_base_est_dag\
                         = gSolver.identify_intervention_utigsp(
                                        intv_args_dict,args["stage2_samples"])
    metric_dict["est_dag"]=est_dag
    metric_dict["oracle_est_dag"]=oracle_est_dag
    metric_dict["igsp_dag"]=igsp_est_dag
    metric_dict["intv_base_est_dag"]=intv_base_est_dag


    #Evaluation: 
    print("==========================")
    print("Estimated Target List")
    for comp in intv_args_dict.keys():
        if "left_comp" in comp:
            print("actual_tgt:{}\test_tgt:{}".format(
                                            comp,
                                            intv_args_dict[comp]["est_tgt"],
                                            )
            )
        elif "est_tgt" in intv_args_dict[comp]:
            print("actual_tgt:{}\test_tgt:{}\toracle_tgt:{}".format(
                                            comp,
                                            intv_args_dict[comp]["est_tgt"],
                                            intv_args_dict[comp]["oracle_est_tgt"]
                                            )
            )
        else:
            #All the targets will have the oracle targets
            print("actual_tgt:{}\toracle_tgt:{}".format(
                                            comp,
                                            intv_args_dict[comp]["oracle_est_tgt"]
                                            )
            )
    #Computing the SHD
    print("==========================")
    metric_dict=compute_shd(intv_args_dict,metric_dict)
    print("SHD:",metric_dict["shd"])
    print("Oracle-SHD:",metric_dict["oracle_shd"])
    print("Intv Base-SHD:",metric_dict["intv_base_shd"])
    print("IGSP-SHD:",metric_dict["igsp_shd"])
    print("actgraph:\n",metric_dict["act_dag"])
    print("est graph:\n",metric_dict["est_dag"])
    print("oracle est graph:\n",metric_dict["oracle_est_dag"])
    print("igsp est graph:\n",metric_dict["igsp_dag"])

    #Computing the js of target
    print("==========================")
    metric_dict=compute_target_jaccard_sim(intv_args_dict,metric_dict,args["dtype"])
    print("Avg JS:",metric_dict["avg_js"])
    print("Avg Oracle JS:",metric_dict["avg_oracle_js"])
    # print("Avg Intv Base JS:",metric_dict["avg_intv_base_js"])
    
    
    #Dumping the experiment
    pickle_experiment_result_json(args,intv_args_dict,metric_dict)
    return intv_args_dict,metric_dict

def compute_shd(intv_args_dict,metric_dict):
    '''
    Assumption: the graph returned by utgsp is a dag (I think it is true
    from the code)
    '''
    #First of all we have to create a DAG object as per the causaldag
    obs_A = intv_args_dict["obs"]["true_params"]["Ai"]
    # print(obs_A)
    num_nodes = obs_A.shape[0]
    #Creating the actual DAG 
    act_dag = cd.DAG()
    act_dag.add_nodes_from([idx for idx in range(num_nodes)])
    #Adding the edges
    for tidx in range(num_nodes):
        for fidx in range(0,num_nodes):
            # print("tidx:{}\tfidx:{}\tval:{}".format(tidx,fidx,obs_A[fidx][tidx]))
            if abs(obs_A[tidx][fidx])>0:
                # print("Adding the edge:{}-->{}",fidx,tidx)
                act_dag.add_arc(fidx,tidx)
    metric_dict["act_dag"]=act_dag

    #Computing the shd with est dag
    est_dag = metric_dict["est_dag"]
    shd = est_dag.shd(act_dag)
    metric_dict["shd"]=shd

    #Computing the SHD for the oracle est dag
    oracle_est_dag = metric_dict["oracle_est_dag"]
    oracle_shd = oracle_est_dag.shd(act_dag)
    metric_dict["oracle_shd"]=oracle_shd

    #Computing the SHD for the intv_base base est dag
    metric_dict["intv_base_shd"]=None
    if metric_dict["intv_base_est_dag"]!=None:
        intv_base_est_dag = metric_dict["intv_base_est_dag"]
        intv_base_shd = intv_base_est_dag.shd(act_dag)
        metric_dict["intv_base_shd"]=intv_base_shd

    #Computing the shd for the oracle igsp dag
    metric_dict["igsp_shd"]=None
    if "igsp_dag" in metric_dict and metric_dict["igsp_dag"]!=None:
        igsp_est_dag = metric_dict["igsp_dag"]
        igsp_shd = igsp_est_dag.shd(act_dag)
        metric_dict["igsp_shd"]=igsp_shd
    
    return metric_dict

def compute_target_jaccard_sim(intv_args_dict,metric_dict,dtype):
    '''
    #Assumption: This assumes that the component is atomic
    right now.
    '''
    similarity_list = []
    oracle_similarity_list = []
    intv_base_similarity_list = []
    for comp in intv_args_dict.keys():
        #We will skip if the componet is the left out one
        if "left" in comp:
            continue

        #Computing the similarity for each component
        if comp=="obs":
            actual_tgt=set([])
        elif dtype=="simulation":
            actual_tgt = set([int(comp)])
        elif dtype=="sachs":
            actual_tgt = set([int(intv_args_dict[comp]["tgt_idx"])])
        else:
            raise NotImplementedError()
        
        
        #Computing the est_target JS
        #It is possible that we have less estimated component thatn actual
        if "est_tgt" in intv_args_dict[comp]:
            est_tgt = set(intv_args_dict[comp]["est_tgt"])
            if comp=="obs" and len(est_tgt.union(actual_tgt))==0:
                js=1.0
            else:
                js = len(est_tgt.intersection(actual_tgt))\
                            /len(est_tgt.union(actual_tgt))
            similarity_list.append(js)

        
        #Computing the oracle JS
        oracle_est_tgt = set(intv_args_dict[comp]["oracle_est_tgt"])
        if comp=="obs" and len(oracle_est_tgt.union(actual_tgt))==0:
            oracle_js=1.0
        else:
            oracle_js = len(oracle_est_tgt.intersection(actual_tgt))\
                        /len(oracle_est_tgt.union(actual_tgt))
        oracle_similarity_list.append(oracle_js)
        
        #Computing the inv base JS
        # intv_base_est_tgt = set(intv_args_dict[comp]["intv_base_est_tgt"])
        # if comp=="obs" and len(intv_base_est_tgt.union(actual_tgt))==0:
        #     intv_base_js=1.0
        # else:
        #     intv_base_js = len(intv_base_est_tgt.intersection(actual_tgt))\
        #                 /len(intv_base_est_tgt.union(actual_tgt))
        # intv_base_similarity_list.append(intv_base_js)

    avg_js = np.mean(similarity_list)
    avg_oracle_js = np.mean(oracle_similarity_list)
    # avg_intv_base_js = np.mean(intv_base_similarity_list)

    metric_dict["avg_js"]=avg_js
    metric_dict["avg_oracle_js"]=avg_oracle_js
    # metric_dict["avg_intv_base_js"]=avg_intv_base_js

    return metric_dict

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj,cd.DAG):
            #Converting the DAG objects to adjmatrix 
            #BEWARE, this adj matrix is in different form 
            #(so use DAG.from_amat to recover graph)
            return dict(
                        amat=obj.to_amat()[0].tolist(),
                        nodes=obj.to_amat()[1],
            )
        return super(NpEncoder, self).default(obj)

def pickle_experiment_result_json(expt_args,intv_args_dict,metric_dict):
    '''
    '''
    #First of all we will remove any samples from this
    comp_list = intv_args_dict.keys()
    for comp in comp_list:
        if "samples" in intv_args_dict[comp]:
            del intv_args_dict[comp]["samples"]

    #Now we are ready to save the results
    experiment_dict = dict(
                        expt_args=expt_args,
                        intv_args_dict=intv_args_dict,
                        metric_dict=metric_dict
    )

    write_fname = "{}/exp_{}.json".format(
                                    expt_args["save_dir"],
                                    expt_args["exp_name"]
    )
    print("Writing the results to: ",write_fname)
    with open(write_fname,"w") as whandle:
        json.dump(experiment_dict,whandle,cls=NpEncoder,indent=4)


#PARLLEL EXPERIMENT RUNNER (SIMULATION)
def jobber(all_expt_config,save_dir,num_parallel_calls):
    '''
    '''
    #First of all we have to generate all possible experiments
    flatten_args_key = []
    flatten_args_val = []
    for key,val in all_expt_config.items():
        flatten_args_key.append(key)
        flatten_args_val.append(val)
    
    #Getting all the porblem configs
    problem_configs = list(it.product(*flatten_args_val))
    #Now generating all the experimetns arg
    all_expt_args = []

    expt_args_list = []
    for cidx,config in enumerate(problem_configs):
        config_dict = {
            key:val for key,val in zip(flatten_args_key,config)
        }

        args = dict(
                save_dir=save_dir,
                exp_name="{}".format(cidx),
                num_nodes = config_dict["num_nodes"],
                num_tgt_prior = config_dict["num_nodes"]+1,
                obs_noise_mean = config_dict["obs_noise_mean"],
                obs_noise_var = config_dict["obs_noise_var"],
                obs_noise_gamma_shape = config_dict["obs_noise_gamma_shape"],
                noise_type = config_dict["noise_type"],
                max_edge_strength = config_dict["max_edge_strength"],
                graph_sparsity_method = config_dict["graph_sparsity_method"],
                adj_dense_prop = config_dict["adj_dense_prop"],
                num_parents = config_dict["num_parents"],
                new_noise_mean = config_dict["new_noise_mean"],
                mix_samples = config_dict["sample_size"],
                stage2_samples = config_dict["sample_size"],
                gmm_tol=config_dict["gmm_tol"],
                intv_type=config_dict["intv_type"],
                new_noise_var=config_dict["new_noise_var"],
                dtype="simulation",
                cutoff_drop_ratio=config_dict["cutoff_drop_ratio"],
        )
        if config_dict["intv_targets"]=="all":
            args["intv_targets"]=list(range(config_dict["num_nodes"]))
        elif config_dict["intv_targets"]=="half":
            #Here we will only interven on half the nodes
            nodes_list = list(range(config_dict["num_nodes"]))
            np.random.shuffle(nodes_list)
            half_node_list = nodes_list[0:config_dict["num_nodes"]//2]
            args["intv_targets"]=half_node_list
        elif type(config_dict["intv_targets"])==type(1):
            nodes_list = list(range(config_dict["num_nodes"]))
            np.random.shuffle(nodes_list)
            some_node_list = nodes_list[0:config_dict["intv_targets"]]
            args["intv_targets"]=some_node_list
        else:
            raise NotImplementedError
        expt_args_list.append(args)
        
        
        #If we want to run sequentially    
        # intv_args_dict,metric_dict = run_mixture_disentangle(args)
        # print("=================================================")
        # print("\n\n\n\n\n\n")
    
    #Running the experiment parallely
    # run_mixture_disentangle(expt_args_list[0])
    with mp.Pool(num_parallel_calls) as p:
        p.map(run_mixture_disentangle,expt_args_list)
    print("Completed the whole experiment!")

def run_simulation_experiments():
    '''
    '''
    # Graphs Related Parameters
    all_expt_config = dict(
        #Graph related parameters
        run_list = list(range(5)), #for random runs with same config, needed?
        num_nodes = [6,],
        max_edge_strength = [1.0,],
        graph_sparsity_method=["adj_dense_prop",],#[adj_dense_prop, use num_parents]
        num_parents = [None],
        adj_dense_prop = [0.1,0.4,0.6,1.0],
        noise_type=["gaussian"], #"gaussian", or gamma
        obs_noise_mean = [0.0],
        obs_noise_var = [1.0],
        obs_noise_gamma_shape = [None],
        #Intervnetion related related parameretrs
        new_noise_mean= [1.0],
        intv_targets = ["half",], #all, half
        intv_type = ["do"], #hard,do,soft
        new_noise_var = [None],
        #Sample and other statistical parameters
        sample_size = [2**idx for idx in range(10,21)],
        gmm_tol = [1e-3], #1e-3 default #10000,5000,1000 for large nodes
        cutoff_drop_ratio=[0.07]
    )


    save_dir="all_expt_logs/expt_logs_sim_compsel_backwardbugfixed_cameraready_half_density_var"
    pathlib.Path(save_dir).mkdir(parents=True,exist_ok=True)
    jobber(all_expt_config,save_dir,num_parallel_calls=1)#64)



#SACHS Dataset
def run_sachs_experiments():
    '''
    '''
    num_parallel_calls=64
    #Setting up the save directory
    save_dir = "all_expt_logs/expt_logs_sachs_compsel_backwardbugfixed_cameraready_temp"
    pathlib.Path(save_dir).mkdir(parents=True,exist_ok=True)

    #Setting up the dataset path and other parameters
    dataset_path="datasets/sachs_yuhaow.csv"
    all_sample_size_factor = [7]
    gmm_tol=1000
    run_list = range(1)
    num_tgt_prior=12
    cutoff_drop_ratio_list=[0.01,]
    


    expt_args_list=[]
    counter=0
    for run_num in run_list:
        for sfactor in all_sample_size_factor:
            for cutoff_drop_ratio in cutoff_drop_ratio_list:
                counter+=1
                config_dict=dict(
                    gmm_tol = gmm_tol,
                    sample_size = int(600*sfactor)
                )
                args = dict(
                            save_dir=save_dir,
                            dataset_path=dataset_path,
                            exp_name="{}".format(counter),
                            # num_nodes = config_dict["num_nodes"],
                            # obs_noise_mean = config_dict["obs_noise_mean"],
                            # obs_noise_var = config_dict["obs_noise_var"],
                            # max_edge_strength = config_dict["max_edge_strength"],
                            # graph_sparsity_method = config_dict["graph_sparsity_method"],
                            # adj_dense_prop = config_dict["adj_dense_prop"],
                            # num_parents = config_dict["num_parents"],
                            # new_noise_mean = config_dict["new_noise_mean"],
                            mix_samples = config_dict["sample_size"],
                            stage2_samples = config_dict["sample_size"],
                            gmm_tol=config_dict["gmm_tol"],
                            num_tgt_prior=num_tgt_prior,
                            cutoff_drop_ratio=cutoff_drop_ratio,
                            # intv_type=config_dict["intv_type"],
                            # new_noise_var=config_dict["new_noise_var"],
                            dtype="sachs"
                )
                expt_args_list.append(args)
    
    # run_mixture_disentangle(expt_args_list[0])
    with mp.Pool(num_parallel_calls) as p:
        p.map(run_mixture_disentangle,expt_args_list)
    print("Completed the whole experiment!")

if __name__=="__main__":
    #If we want to run the simulation experiments then we will open this
    run_simulation_experiments()

    #If we want to run the resutls on the SACHS dataset
    # run_sachs_experiments()
    
    