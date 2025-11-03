import os
import sys
from collections import defaultdict
from statistics import mean, stdev, variance
import numpy as np
import pandas as pd

from src.coco.util.utils import f1_score
from src.exp.exp_coco.method_types import MethodType



class ResultsCoCo:
    def __init__(self, writing):
        self.eval_pairs = defaultdict(defaultdict) #confounded edges
        self.eval_causal = defaultdict(defaultdict) #causal edges
        self.eval_components = defaultdict(defaultdict) #clustering metrics

        self.result_pairs = {}
        self.result_causal = {}
        self.result_components = {}

        self.writing = writing

    def _write(self, path, st):
        if self.writing:
            write_file = open(path, "a+")
            write_file.write(st)
            write_file.close()
        else:
            sys.stdout.write(st)

    def get_key(self, NO, NCO, N,  NAstar, NC):
        return str(NO) + '_' + str(NCO) + '_' + str(N) + '_' + str(NAstar) + '_' + str(NC)

    def init_entry(self, NO, NCO, N, NA, NC, metric, method):
        key = self.get_key(NO, NCO, N, NA, NC)

        for dic in [self.eval_causal, self.eval_pairs, self.eval_components]:
            if not dic[key].__contains__(metric):
                dic[key][metric] = {}

        if not self.eval_pairs[key][metric].__contains__(method):
            self.eval_pairs[key][metric][method] = pd.DataFrame(
                {'tp': [], 'fp': [], 'tn': [], 'fn': [],
                 'f1': [], 'tpr': [], 'fpr': []})

        if not self.eval_causal[key][metric].__contains__(method):
            self.eval_causal[key][metric][method] = pd.DataFrame(
                {'tp': [], 'fp': [], 'tn': [], 'fn': [],
                 'f1': [], 'fpr': [], 'tpr': [],
                 'tptp': [], 'tpfp': [], 'tptn': [], 'tpfn': [],
                 'fptp': [], 'fpfp': [], 'fptn': [], 'fpfn': [],
                 'tntp': [], 'tnfp': [], 'tntn': [], 'tnfn': [],
                 'fntp': [], 'fnfp': [], 'fntn': [], 'fnfn': []})

        if not self.eval_components[key][metric].__contains__(method):
            self.eval_components[key][metric][method] = pd.DataFrame(
                {'jacc': [], 'ari': [], 'ami': [], 'tp': [], 'fp': [], 'tn': [], 'fn': [], 'f1': [],
                 'tp-adj': [], 'fp-adj': [], 'tn-adj': [], 'fn-adj': [], 'f1-adj': []})

    def update(self, dag, n_nodes, n_confounders, n_contexts, n_shifts_observed, n_shifts_confounders, coco_results, methods):

        key = self.get_key(n_nodes, n_confounders, n_contexts, n_shifts_observed, n_shifts_confounders)

        for metric in coco_results.keys():
            for method in methods:
                self.init_entry(n_nodes, n_confounders, n_contexts, n_shifts_observed, n_shifts_confounders, metric, method)
                caus = 0,0,0,0,0
                if method.is_coco():
                    if method == MethodType.ORACLE:
                        pairs = coco_results[metric][str(method)].eval_oracle_edges(dag)
                        components = coco_results[metric][str(method)].eval_oracle_graph_cuts(dag)
                    else:
                        pairs = coco_results[metric][str(method)].eval_estimated_edges(dag)
                        components = coco_results[metric][str(method)].eval_estimated_graph_cuts(dag)
                    caus = coco_results[metric][str(method)].eval_causal(dag)
                else:
                    if method.value == MethodType.MSS.value:
                        components = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
                        pairs = coco_results[metric][str(method)].eval_estimated_edges(dag)  #this should return only tn and fn
                        caus = coco_results[metric][str(method)].eval_causal(dag)
                    else:
                        assert (method.is_fci())
                        components = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
                        tp, fp, tn, fn, f1 = coco_results[metric][str(method)].eval_confounded(dag, method)
                        pairs = (tp, fp, tn, fn, f1, 0, 0)
                        tp, fp, tn, fn, f1 = coco_results[metric][str(method)].eval_causal(dag, method)
                        caus = (tp, fp, tn, fn, f1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)

                self.eval_pairs[key][metric][method].loc[len(self.eval_pairs[key][metric][method])] = pairs
                self.eval_causal[key][metric][method].loc[len(self.eval_causal[key][metric][method])] = caus
                self.eval_components[key][metric][method].loc[len(self.eval_components[key][metric][method])] = components

    def get_result(self, metric, method=MethodType.ORACLE):
        for key in self.eval_causal:
            if not self.result_causal.__contains__(key):
                self.result_causal[key] = {}
            if not self.result_causal[key].__contains__(metric):
                self.result_causal[key][metric] = {}

            N = len(self.eval_causal[key][metric][method]['tp'])
            tpc = sum(self.eval_causal[key][metric][method]['tp'])
            fpc = sum(self.eval_causal[key][metric][method]['fp'])
            tnc = sum(self.eval_causal[key][metric][method]['tn'])
            fnc = sum(self.eval_causal[key][metric][method]['fn'])
            f1c = sum(self.eval_causal[key][metric][method]['f1']) / N
            f1stdevc = stdev(self.eval_causal[key][metric][method]['f1'])
            f1sigc = variance(self.eval_causal[key][metric][method]['f1'])

            tprc = sum(self.eval_causal[key][metric][method]['tpr']) / N
            tprstdevc = stdev(self.eval_causal[key][metric][method]['tpr'])
            tprsigc = variance(self.eval_causal[key][metric][method]['tpr'])

            fprc = sum(self.eval_causal[key][metric][method]['fpr']) / N
            fprstdevc = stdev(self.eval_causal[key][metric][method]['fpr'])
            fprsigc = variance(self.eval_causal[key][metric][method]['fpr'])

            self.result_causal[key][metric][method] =\
                {'tp': tpc, 'fp': fpc, 'tn': tnc, 'fn': fnc,
                 'f1': f1c, 'f1sig': f1sigc, 'f1stdev': f1stdevc,
                 'tpr': tprc, 'tprsig': tprsigc, 'tprstdev': tprstdevc,
                 'fpr': fprc, 'fprsig': fprsigc, 'fprstdev': fprstdevc
                }

        for key in self.eval_components:
            if not self.result_components.__contains__(key):
                self.result_components[key] = {}
            if not self.result_components[key].__contains__(metric):
                self.result_components[key][metric] = {}
            N = len(self.eval_components[key][metric][method]['ari'])
            tp = sum(self.eval_components[key][metric][method]['tp'])
            fp = sum(self.eval_components[key][metric][method]['fp'])
            tn = sum(self.eval_components[key][metric][method]['tn'])
            fn = sum(self.eval_components[key][metric][method]['fn'])
            f1 = sum(self.eval_components[key][metric][method]['f1'])/N
            f1stdev = stdev(self.eval_components[key][metric][method]['f1'])
            f1sig = variance(self.eval_components[key][metric][method]['f1'])

            tpa = sum(self.eval_components[key][metric][method]['tp-adj'])
            fpa = sum(self.eval_components[key][metric][method]['fp-adj'])
            tna = sum(self.eval_components[key][metric][method]['tn-adj'])
            fna = sum(self.eval_components[key][metric][method]['fn-adj'])
            f1a = sum(self.eval_components[key][metric][method]['f1-adj'])/N

            f1astdev = stdev(self.eval_components[key][metric][method]['f1-adj'])
            f1asig = variance(self.eval_components[key][metric][method]['f1-adj'])

            jacc = mean(self.eval_components[key][metric][method]['jacc'])
            ari = mean(self.eval_components[key][metric][method]['ari'])
            ami = mean(self.eval_components[key][metric][method]['ami'])


            self.result_components[key][metric][method] =\
                { 'jacc': jacc, 'ari': ari, 'ami': ami, 'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn,
                  'f1': f1, 'f1stdev' : f1stdev, 'f1sig': f1sig,
                 'tp-adj': tpa, 'fp-adj': fpa, 'tn-adj': tna, 'fn-adj': fna,
                  'f1-adj': f1a, 'f1stdev-adj' : f1astdev, 'f1sig-adj': f1asig
                  }

        for key in self.eval_pairs:
            if not self.result_pairs.__contains__(key):
                self.result_pairs[key] = {}
            if not self.result_pairs[key].__contains__(metric):
                self.result_pairs[key][metric] = {}
            tp = sum(self.eval_pairs[key][metric][method]['tp'])
            fp = sum(self.eval_pairs[key][metric][method]['fp'])
            tn = sum(self.eval_pairs[key][metric][method]['tn'])
            fn = sum(self.eval_pairs[key][metric][method]['fn'])

            N = len(self.eval_pairs[key][metric][method]['f1'])
            f1 = sum(self.eval_pairs[key][metric][method]['f1'])/N
            f1stdev = stdev(self.eval_pairs[key][metric][method]['f1'])
            f1sig = variance(self.eval_pairs[key][metric][method]['f1'])

            tpr = sum(self.eval_pairs[key][metric][method]['tpr']) / N
            fpr = sum(self.eval_pairs[key][metric][method]['fpr']) / N
            tprstdev = stdev(self.eval_pairs[key][metric][method]['tpr'])
            tprsig = variance(self.eval_pairs[key][metric][method]['tpr'])
            fprstdev = stdev(self.eval_pairs[key][metric][method]['fpr'])
            fprsig = variance(self.eval_pairs[key][metric][method]['fpr'])

            self.result_pairs[key][metric][method] = {'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn,
                                                      'f1': f1, 'f1stdev': f1stdev, 'f1sig': f1sig ,
                                                      'tpr': tpr, 'tprsig': tprsig, 'tprstdev': tprstdev,
                                                      'fpr': fpr, 'fprsig': fprsig, 'fprstdev': fprstdev
                                                      }

    def write(self, write_file, n_nodes, n_confounders, n_contexts, n_shifts_observed, n_shifts_confounders,  metrics,
             methods ):

        key = self.get_key(n_nodes, n_confounders, n_contexts, n_shifts_observed, n_shifts_confounders)
        self._write(write_file, "\n"+key)

        for metric in metrics:
            self._write(write_file, "\n\tMETRIC:"+metric)
            for method in methods:
                self.get_result(metric, method)
                self._write(write_file, "\n\t\t"+str(method)+":\t\t\t")

                for i in ['tp','fp', 'tn', 'fn', 'f1', 'f1stdev', 'f1sig']:
                    if self.result_pairs[key][metric][method].__contains__(i):
                        self._write(write_file, "[" + str(i) +"-pair] " + str(np.round(self.result_pairs[key][metric][method][i], 2)) + "\t")
                self._write(write_file, "\n\t\t\t\t\t\t")
                for i in ['tp','fp', 'tn', 'fn', 'f1', 'f1stdev', 'f1sig']:
                    if self.result_causal[key][metric][method].__contains__(i):
                        self._write(write_file, "[" + str(i) +"-cau] " + str(np.round(self.result_causal[key][metric][method][i], 2)) + "\t")
                self._write(write_file, "\n\t\t\t\t\t\t")
                for i in ['tp-adj', 'fp-adj', 'tn-adj', 'fn-adj', 'f1-adj',  'f1stdev-adj', 'f1sig-adj' ]:
                    if self.result_components[key][metric][method].__contains__(i):
                        self._write(write_file, "[" + str(i) +"] " + str(np.round(self.result_components[key][metric][method][i], 2)) + "\t")
                self._write(write_file, "\n\t\t\t\t\t\t")
                for i in ['tp', 'fp', 'tn', 'fn', 'f1', 'f1stdev', 'f1sig', 'jacc', 'ari', 'ami']: #, 'f1c']:
                    if self.result_components[key][metric][method].__contains__(i):
                        self._write(write_file, "[" + str(i) +"-clu] " + str(np.round(self.result_components[key][metric][method][i], 2)) + "\t")


    def write_final(self, write_file, metrics, methods):
        for method in methods:
            for metric in metrics:
                self.get_result(metric, method)
            self._write(write_file, "\n\n"+str(method)+"\nCONTEXTS_OBSSHIFT_CFDSHIFT")

            for metric in metrics:

                TP, FP, TN, FN = 0, 0, 0, 0
                TPA, FPA, TNA, FNA = 0, 0, 0, 0
                F1A, F1,F1stdev,F1sig, F12, F12stdev,F12sig, F1Astdev, F1Asig, F1_caus, F1sig_caus, F1stdev_caus = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
                TP_pairs, FP_pairs, TN_pairs, FN_pairs = 0, 0, 0, 0
                TP_caus, FP_caus, TN_caus, FN_caus = 0, 0, 0, 0
                JACC, ARI, AMI, ct = 0, 0, 0, 0
                self._write(write_file, "\n\tMETRIC:" + metric )
                for key in self.result_pairs:
                    self._write(write_file, "\n\t\t\t"+key+"\t")
                    self._write(write_file, f"\t\t[f1-p] {str(np.round(self.result_pairs[key][metric][method]['f1'], 2))} \t+-{str(np.round(self.result_pairs[key][metric][method]['f1stdev'], 2))} ({str(np.round(self.result_pairs[key][metric][method]['f1sig'], 2))}) \t")
                    self._write(write_file,
                        f"\t\t[f1-cau] {str(np.round(self.result_causal[key][metric][method]['f1'], 2))} \t+-{str(np.round(self.result_causal[key][metric][method]['f1stdev'], 2))} ({str(np.round(self.result_causal[key][metric][method]['f1sig'], 2))}) \t")
                    self._write(write_file,
                        f"\t\t[f1-clu] {str(np.round(self.result_components[key][metric][method]['f1'], 2))} \t+-{str(np.round(self.result_components[key][metric][method]['f1stdev'], 2))} ({str(np.round(self.result_components[key][metric][method]['f1sig'], 2))}) \t")
                    self._write(write_file,
                        f"\t\t[jacc] {str(np.round(self.result_components[key][metric][method]['jacc'], 2))} \t")
                    self._write(write_file,
                        f"\t\t[f1-a] {str(np.round(self.result_components[key][metric][method]['f1-adj'], 2))} \t+-{str(np.round(self.result_components[key][metric][method]['f1stdev-adj'], 2))} ({str(np.round(self.result_components[key][metric][method]['f1sig-adj'], 2))}) \t")

                    # TODO mess
                    TP_pairs += self.result_pairs[key][metric][method]['tp']
                    FP_pairs += self.result_pairs[key][metric][method]['fp']
                    TN_pairs += self.result_pairs[key][metric][method]['tn']
                    FN_pairs += self.result_pairs[key][metric][method]['fn']
                    F1 += self.result_pairs[key][metric][method]['f1']
                    F1sig += self.result_pairs[key][metric][method]['f1sig']
                    F1stdev += self.result_pairs[key][metric][method]['f1stdev']

                    TP_caus += self.result_causal[key][metric][method]['tp']
                    FP_caus += self.result_causal[key][metric][method]['fp']
                    TN_caus += self.result_causal[key][metric][method]['tn']
                    FN_caus += self.result_causal[key][metric][method]['fn']
                    F1_caus += self.result_causal[key][metric][method]['f1']
                    F1sig_caus += self.result_causal[key][metric][method]['f1sig']
                    F1stdev_caus += self.result_causal[key][metric][method]['f1stdev']

                    TP += self.result_components[key][metric][method]['tp']
                    FP += self.result_components[key][metric][method]['fp']
                    TN += self.result_components[key][metric][method]['tn']
                    FN += self.result_components[key][metric][method]['fn']
                    F12 += self.result_components[key][metric][method]['f1']
                    F12sig += self.result_components[key][metric][method]['f1sig']
                    F12stdev += self.result_components[key][metric][method]['f1stdev']

                    TPA += self.result_components[key][metric][method]['tp-adj']
                    FPA += self.result_components[key][metric][method]['fp-adj']
                    TNA += self.result_components[key][metric][method]['tn-adj']
                    FNA += self.result_components[key][metric][method]['fn-adj']
                    F1A += self.result_components[key][metric][method]['f1-adj']
                    F1Asig += self.result_components[key][metric][method]['f1sig-adj']
                    F1Astdev += self.result_components[key][metric][method]['f1stdev-adj']

                    JACC += self.result_components[key][metric][method]['jacc']
                    ARI += self.result_components[key][metric][method]['ari']
                    AMI += self.result_components[key][metric][method]['ami']
                    ct = ct + 1
                self._write(write_file, "\n\tOVERALL\n\t\tx_x_x\t[tp-pairs] " + str(TP_pairs) + "\t[fp-pairs] " + str(FP_pairs) + "\t[tn-pairs] " + str(TN_pairs) + "\t[fn-pairs] " + str(FN_pairs) + "\t[f1-pairs] " + str(round(f1_score(TP_pairs, FP_pairs, FN_pairs), 2))+ "\t+-" + str(np.round(F1sig, 2))+ "(" + str(np.round(F1stdev, 2)) + ")"
                + "\n\t\t\t\t[tp-adj] " + str(TPA) + "\t[fp-adj] " + str(FPA) + "\t[tn-adj] " + str(TNA) + "\t[fn-adj] " + str(FNA) + "\t[f1-adj] " + str(np.round(f1_score(TPA, FPA, FNA), 2)) + "\t+-" + str(np.round(F1Asig, 2))+ "("  + str(np.round(F1Astdev, 2))+ ")"
                                 + "\n\t\t\t\t[tp-caus] " + str(TP_caus) + "\t[fp-caus] " + str(FP_caus) + "\t[tn-caus] " + str(
                    TN_caus) + "\t[fn-caus] " + str(FN_caus) + "\t[f1-caus] " + str(np.round(f1_score(TP_caus, FP_caus, FN_caus), 2)) + "\t+-" + str(
                    np.round(F12sig, 2)) + "(" + str(np.round(F12stdev, 2)) + ")"
                + "\n\t\t\t\t[tp] " + str(TP)  + "\t[fp] " + str(FP)  + "\t[tn] " + str(TN) + "\t[fn] " + str(FN) + "\t[f1] " + str(np.round(f1_score(TP, FP, FN),2))+ "\t+-" + str(np.round(F12sig, 2))+ "(" + str(np.round(F12stdev, 2))+ ")"
                                 + "\n\t\t\t\t[jacc] "+ str(round(JACC/ct,2)) + "\t[ari] "+ str(round(ARI/ct,2)) + "\t[ami] "+ str(round(AMI/ct,2)) )






    def write_methods_tex(self, identifier, path,
                          test_cases_entry,
                          methods=[m for m in MethodType],
                          fscore = 'f1',
                          sigscore = 'f1sig',
                          metric='mi'):
        ''' Generate a table used in tex figures,results per method with one varying variable. Y/rows: methods, X/cols: test cases varying one variabe.

        :param identifier: pattern (nx, nconf, ncontexts, sx, sz)
        :param path: path
        :param test_cases_entry: test cases, for example {(nx,..,sx,1), (nx,..,sx,2), (nx,..,sx,3)}
        :param methods: [MethodType.xy]
        :param fscore: main score evaluated
        :param sigscore: conf column
        :param metric: mi
        :return:
        '''
        path = f"{path}/tex_coco/"
        if self.writing and not os.path.exists(path):
            os.makedirs(path)

        show = ""
        for causal in [True, False]:
            if causal:
                path= f"{path}tex_causal_{fscore}_{identifier}.csv"
                show += '\nCAUSAL_DIRECTIONS\n'
            else:
                path = f"{path}tex_confounded_{fscore}_{identifier}.csv"
                show += '\nCONFOUNDED_PAIRS\n'
            if self.writing:
                self._write(path, 'X')
            show += 'X'

            for entry in methods:
                self._write(path, '\t'+str(entry)+'\t'+str(entry)+"_conf")
                show += str('\t'+str(entry)+'\t'+str(entry)+"_conf")
            for xnm in test_cases_entry:
                (n_nodes, n_confounders, n_contexts, n_shifts_observed, n_shifts_confounders, nm) = test_cases_entry[str(xnm)]
                key = self.get_key(n_nodes, n_confounders, n_contexts, n_shifts_observed, n_shifts_confounders)

                self._write(path, '\n'+str(xnm))
                show+='\n'+str(nm)+str(xnm)
                for method in methods:
                    if causal:
                        res = self.result_causal
                    else:
                        res = self.result_pairs
                    self._write(path,
                        f"\t{str(np.round(res[key][metric][method][fscore], 2))}\t{str(np.round(res[key][metric][method][sigscore], 2))}")
                    show += f"\t{str(np.round(res[key][metric][method][fscore], 2))}\t{str(np.round(res[key][metric][method][sigscore], 2))}"

        return show

    def write_keys_tex(self, identifier, path, test_cases, x_names, method=MethodType.ORACLE,
                       fscore='f1', sigscore='f1sig', metric='mi'):
        ''' Generate a table for tex figs, showing results for one method and two varying variables.

        :param identifier: for file
        :param path: path
        :param test_cases: for example {ynm1: {xnm1: (nx,..,a,1),xnm2: (nx,..,b,1)} ynm2: {xnm1: (nx,..,a,2),xnm2: (nx,..,b,2)} }
        :param x_names: rows, xnm1, xnm2
        :param method: method evaluated
        :param fscore: main score evaluated
        :param sigscore: conf column
        :param metric: mi
        :return:
        '''
        path = f"{path}/tex_coco/"
        if self.writing and not os.path.exists(path):
            os.makedirs(path)

        show = ""
        for causal in [True, False]:
            if causal:
                path = f"{path}tex_causal_{fscore}_{identifier}.csv"
                show += '\nCAUSAL_DIRECTIONS\n'
            else:
                path = f"{path}tex_confounded_{fscore}_{identifier}.csv"
                show += '\nCONFOUNDED_PAIRS\n'
            self._write(path, 'X')
            show += 'X'
            for entry in test_cases:
                self._write(path, '\t'+entry+'\t'+entry+"_conf")
                show += str('\t\t'+entry+'\t'+entry+"_conf")
            for xnm in x_names:
                self._write(path, '\n'+str(xnm))
                show +='\n'+str(xnm)
                for case_key in test_cases:

                    if str(xnm) in test_cases[case_key]:
                            (n_nodes, n_confounders, n_contexts, n_shifts_observed, n_shifts_confounders, nm) = test_cases[case_key][str(xnm)]
                            key = self.get_key(n_nodes, n_confounders, n_contexts, n_shifts_observed, n_shifts_confounders)

                            if causal:
                                res = self.result_causal
                            else:
                                res = self.result_pairs
                            self._write(path,
                                f"\t{str(np.round(res[key][metric][method][fscore], 2))}\t{str(np.round(res[key][metric][method][sigscore], 2))}")
                            show += f"\t{key}:\t{str(np.round(res[key][metric][method][fscore], 2))}\t{str(np.round(res[key][metric][method][sigscore], 2))}"

        return show
