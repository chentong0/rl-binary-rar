# get K median and K max
from collections import defaultdict
from statistics import median

def get_stats(domain_model_triplet_dict):
    domain_K_dict = defaultdict(lambda: defaultdict(int))
    for domain, model_triplet_dict in domain_model_triplet_dict.items():
        claim_num_lst = []
        for model_name, triplet_lst in model_triplet_dict.items():
            for triplet in triplet_lst:
                claim_num_lst.append(triplet[1])

        claim_num_lst.sort()
        K_median = claim_num_lst[len(claim_num_lst)//2]
        K_max = claim_num_lst[-1]
        domain_K_dict[domain]["K_median"] = K_median
        domain_K_dict[domain]["K_max"] = K_max
        # print(f"{domain} - {K_median}: {K_max}")

    return domain_K_dict

def get_avg_numbers(domain_model_triplet_dict, domain_K_dict):
    results = []
    domain_to_K = {"LongFacts": 32, "Factscore": 26, "ELI5": 21, "AskHistorian": 21, "new_books": 24, "ShareGPT": 11, "Factscore_train_template": 10}

    for domain, model_triplet_dict in domain_model_triplet_dict.items():
        K_median = domain_K_dict[domain]["K_median"]
        K_max = domain_K_dict[domain]["K_max"]
        K_ref = domain_to_K.get(domain, None)
        # assert K_ref is not None, f"Domain {domain} not found in domain_to_K mapping."
        # warning if K_ref is not found
        if K_ref is None:
            print(f"Warning: Domain {domain} not found in domain_to_K mapping.")

        table_content = []
        F1_at_median_lst = []
        for model_name in model_triplet_dict.keys():
            triplet_lst = domain_model_triplet_dict[domain][model_name]

            sent_len_lst = [x[2] for x in triplet_lst]
            sup_lst = [x[0] for x in triplet_lst]
            uns_lst = [x[1] - x[0] for x in triplet_lst]
            prec_lst = [x[0] / x[1] if x[1] > 0 else 1.0 for x in triplet_lst]
            rec_med_lst = [min(x[0] / K_median, 1) for x in triplet_lst] if K_median else [0] * len(triplet_lst)
            rec_max_lst = [min(x[0] / K_max, 1) for x in triplet_lst] if K_max else [0] * len(triplet_lst)
            rec_ref_lst = [min(x[0] / K_ref, 1) for x in triplet_lst] if K_ref else [0] * len(triplet_lst)

            # get f1@K median and f1@K max
            f1_med_lst = [2 * prec * rec_med / (prec + rec_med) if rec_med > 0 else 0 for prec, rec_med in zip(prec_lst, rec_med_lst)]
            f1_max_lst = [2 * prec * rec_max / (prec + rec_max) if rec_max > 0 else 0 for prec, rec_max in zip(prec_lst, rec_max_lst)]
            f1_ref_lst = [2 * prec * rec_ref / (prec + rec_ref) if rec_ref > 0 else 0 for prec, rec_ref in zip(prec_lst, rec_ref_lst)]

            # get ave. numbers
            ave_sent = sum(sent_len_lst) / len(sent_len_lst)
            S = sum(sup_lst) / len(sup_lst)
            U = sum(uns_lst) / len(uns_lst)
            P = sum(prec_lst) / len(prec_lst)
            Rec_med = sum(rec_med_lst) / len(rec_med_lst) if K_median else 0
            Rec_max = sum(rec_max_lst) / len(rec_max_lst) if K_max else 0
            Rec_ref = sum(rec_ref_lst) / len(rec_ref_lst) if K_ref else 0
            F1_med = sum(f1_med_lst) / len(f1_med_lst) if K_median else 0
            F1_max = sum(f1_max_lst) / len(f1_max_lst) if K_max else 0
            F1_ref = sum(f1_ref_lst) / len(f1_ref_lst) if K_ref else 0

            table_row = [model_name, domain, round(ave_sent, 3), round(S, 3), round(U, 3), round(P, 3), round(Rec_med, 3), round(Rec_max, 3), round(F1_med, 3), round(F1_max, 3), round(F1_ref, 3)]
            table_content.append(table_row)

            F1_at_median_lst.append(100*round(F1_med, 3))

            print(f"[{domain}-{model_name}] \nF1@k median: {F1_med:.3f}, F1@k max: {F1_max:.3f}")

            results.append({
                "model_name": model_name,
                "domain": domain,
                "ave_sent": round(ave_sent, 3),
                "S": round(S, 3),
                "U": round(U, 3),
                "P": round(P, 3),
                "Rec_med": round(Rec_med, 3),
                "Rec_max": round(Rec_max, 3),
                "Rec_ref": round(Rec_ref, 3),
                "F1_med": round(F1_med, 3),
                "F1_max": round(F1_max, 3),
                "F1_ref": round(F1_ref, 3),
            })
        
    return results


def get_veriscore(domain_model_triplet_dict):
    domain_K_dict= get_stats(domain_model_triplet_dict)
    results = get_avg_numbers(domain_model_triplet_dict, domain_K_dict)
    return results