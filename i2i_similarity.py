def get_sim_item(df, user_col, item_col, use_iif=True): 
    user_item_ = df.groupby(user_col)[item_col].agg(list).reset_index()
    user_item_dict = dict(zip(user_item_[user_col], user_item_[item_col]))
    
    user_time_ = df.groupby(user_col)['unix_time'].agg(list).reset_index() # 引入时间因素
    user_time_dict = dict(zip(user_time_[user_col], user_time_['unix_time']))
    
    sim_item = {}  
    item_cnt = defaultdict(int)  # 商品被点击次数
    for user, items in tqdm(user_item_dict.items()):
        for loc1, item in enumerate(items):  
            item_cnt[item] += 1  
            sim_item.setdefault(item, {})  
            for loc2, relate_item in enumerate(items):  
                if item == relate_item:  # 会过滤点击1次的用户
                    continue  
                t1 = user_time_dict[user][loc1] # 点击时间提取
                t2 = user_time_dict[user][loc2]
                sim_item[item].setdefault(relate_item, 0)  
                loc_alpha = 1 if loc2 > loc1 else 0.8
                loc_weight = loc_alpha * (0.8 ** (np.abs(loc2-loc1) - 1))
                time_weight = np.exp(-np.abs(t2-t1)/60)
                sim_item[item][relate_item] += loc_weight * time_weight / math.log(1 + len(items))

    sim_item_corr = sim_item.copy() # 引入AB的各种被点击次数  
    for i, related_items in tqdm(sim_item.items()):  
        for j, cij in related_items.items():  
            sim_item_corr[i][j] = cij / ((item_cnt[i] * item_cnt[j]) ** 0.2)  
  
    return sim_item_corr, user_item_dict 
