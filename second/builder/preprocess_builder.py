import second.core.preprocess as prep

def build_db_preprocess(db_prep_config):
    prep_type = db_prep_config.WhichOneof('database_preprocessing_step')

    if prep_type == 'filter_by_difficulty':
        cfg = db_prep_config.filter_by_difficulty
        return prep.DBFilterByDifficulty(list(cfg.removed_difficulties))
    elif prep_type == 'filter_by_min_num_points':
        cfg = db_prep_config.filter_by_min_num_points
        return prep.DBFilterByMinNumPoint(dict(cfg.min_num_point_pairs))
    else:
        raise ValueError("unknown database prep type")

