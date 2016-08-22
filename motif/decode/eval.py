
#TODO: actually write this
def score():
    # Generate melody output using predictions
    print "Generating Test Melodies"
    mel_test_dict = {}
    for key in test_contour_dict.keys():
        print key
        mel_test_dict[key] = gm.melody_from_clf(test_contour_dict[key],
                                                prob_thresh=best_thresh,
                                                method=decode)
        fpath = os.path.join(meldir, "%s_pred.csv" % key)
        mel_test_dict[key].to_csv(fpath, header=False, index=True)

    # Score Melody Output
    mel_scores = gm.score_melodies(mel_test_dict, test_annot_dict)

    overall_scores = \
        pd.DataFrame(columns=['VR', 'VFA', 'RPA', 'RCA', 'OA'],
                     index=mel_scores.keys())
    overall_scores['VR'] = \
        [mel_scores[key]['Voicing Recall'] for key in mel_scores.keys()]
    overall_scores['VFA'] = \
        [mel_scores[key]['Voicing False Alarm'] for key in mel_scores.keys()]
    overall_scores['RPA'] = \
        [mel_scores[key]['Raw Pitch Accuracy'] for key in mel_scores.keys()]
    overall_scores['RCA'] = \
        [mel_scores[key]['Raw Chroma Accuracy'] for key in mel_scores.keys()]
    overall_scores['OA'] = \
        [mel_scores[key]['Overall Accuracy'] for key in mel_scores.keys()]

    scores_fpath = os.path.join(outdir, "all_mel_scores.csv")
    overall_scores.to_csv(scores_fpath)

    score_summary = os.path.join(outdir, "mel_score_summary.csv")
    overall_scores.describe().to_csv(score_summary)