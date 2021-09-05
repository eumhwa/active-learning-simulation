import os, random
import numpy as np
import pandas as pd


class ActiveLearning:
    def __init__(self, p_list, features, threshold, sampling_rate, k=20):
        """
        p_list: 2d-list type, ex) [[0.7, 0.2, 0.1], [0.6, 0.2, 0.2], ...]
        features: encoded feature numpy array -> shape: (n_data, n_features)
        sampling_rate: desired number of sampled data
        threshold: entropy threshold
        """
        self.p_list = p_list
        self.feature_np = features
        self.threshold = threshold
        self.sampling_rate = sampling_rate
        self.core_set_k = k
        self.etp_idx_list = []
        self.css_idx_list = []

        self.n_data = len(p_list) if len(p_list) > 0 else features.shape[0]
        self.n_features = features.shape[1]
        self.n_samples = int(self.n_data * self.sampling_rate * 100)

    # -- Entropy sampling ftns
    def H(self, p):
        etp = -np.sum(p * np.log(p)) if p != 1 else 0
        return etp

    def cal_entropy(self, prob_list):
        out = []
        for p in prob_list:
            etp = self.H(p)
            out.append(etp)
        return out

    def entropy_sampling(self):
        etps = np.array(self.cal_entropy(self.p_list))
        top_n_idx = np.argsort(etps)[::-1][: self.n_samples].tolist()

        filtered_top_n_idx = [i for i in top_n_idx if etps[i] >= self.threshold]
        return filtered_top_n_idx

    # -- core set selection ftns
    def euc_dist(self, l, r):
        return np.square(np.sum(l - r))

    def initial_point_sampling(self):
        first_point = random.sample(range(self.n_data), 1)[0]
        self.css_idx_list.append(first_point)

        dist = []
        for i in range(self.n_data):
            tmp_dist = self.euc_dist(self.feature_np[first_point], self.feature_np[i])
            dist.append(tmp_dist)

        second_point = dist.index(max(dist))
        self.css_idx_list.append(second_point)
        print(f"initial points are {self.css_idx_list}")
        return

    def store_distance(self):
        dist_store = np.zeros((self.n_data, 2))
        for p in range(dist_store.shape[0]):
            if p in self.css_idx_list:
                dist_store[p] = [10e5, p]
                continue

            close_dist = 10e5
            close_to = -1
            for q in self.css_idx_list:
                tmp_dist = self.euc_dist(self.feature_np[q], self.feature_np[p])
                if tmp_dist <= close_dist:
                    close_to = q
                    close_dist = tmp_dist

            dist_store[p] = [close_dist, close_to]

        return dist_store

    def append_center_point(self, i):

        dist_store = self.store_distance()
        dist_df = pd.DataFrame(dist_store)
        dist_df.columns = ["dist", "cluster_idx"]

        grouped = dist_df.groupby("cluster_idx")

        cur_idx_size = len(self.css_idx_list)
        for k in range(cur_idx_size):
            new_id = grouped.idxmin().iloc[k].item()
            if new_id not in self.css_idx_list:
                self.css_idx_list.append(new_id)

        print(f"# {i}-th iteration end / current points are {self.css_idx_list}")
        return

    def core_set_selection(self):
        iter = 1
        self.initial_point_sampling()
        while len(self.css_idx_list) < self.core_set_k:
            self.append_center_point(iter)
            iter += 1

        result = sorted(self.css_idx_list)
        return result

    def random_sampling(self, n_pop, n_samp):
        #n_samples_rs = self.core_set_k + self.n_samples
        return random.sample(range(n_pop), n_samp)