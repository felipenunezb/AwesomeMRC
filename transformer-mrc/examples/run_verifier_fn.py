# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Run BERT on SQuAD."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import logging
import collections
import json
import os

from evaluate_official2 import eval_squad

logger = logging.getLogger(__name__)

def get_score1(args):
    cof = [args.cls_beta, 1-args.cls_beta]
    best_cof = [1]
    
    assert args.na_lambda < 1
    
    if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)
    
    #CLS Scores
    all_scores = collections.OrderedDict()
    idx = 0
    for input_file in args.input_cls_files.split(","):
        with open(input_file, 'r') as reader:
            input_data = json.load(reader, strict=False)
            for (key, score) in input_data.items():
                if key not in all_scores:
                    all_scores[key] = []
                all_scores[key].append(cof[idx] * score)
        idx += 1
    cls_score = {}
    for (key, scores) in all_scores.items():
        mean_score = 0.0
        for score in scores:
            mean_score += score
        mean_score /= float(len(scores))
        cls_score[key] = mean_score

    #Max Logit Scores
    idx = 0
    all_nbest = collections.OrderedDict()
    for input_file in args.input_nbest_files.split(","):
        with open(input_file, "r") as reader:
            input_data = json.load(reader, strict=False)
            for (key, entries) in input_data.items():
                if key not in all_nbest:
                    all_nbest[key] = collections.defaultdict(float)
                for entry in entries:
                    all_nbest[key][entry["text"]] += best_cof[idx] * (entry["start_logit"] + entry["end_logit"])
        idx += 1
    output_predictions = {}
    max_logits = {}
    for (key, entry_map) in all_nbest.items():
        sorted_texts = sorted(
            entry_map.keys(), key=lambda x: entry_map[x], reverse=True)
        best_text = sorted_texts[0]
        output_predictions[key] = best_text
        max_logits[key] = entry_map[best_text]
        
    #Null Logit Scores
    idx = 0
    null_odds = collections.OrderedDict()
    for input_file in args.input_av_null.split(","):
        with open(input_file, "r") as reader:
            input_data = json.load(reader, strict=False)
            for (key, score) in input_data.items():
                null_odds[key] = cof[idx] * score

    #Out Score > Score NA - Score HAS
    #Score NA > l1 * null_odds + l2 * cls_score
    output_scores = collections.OrderedDict()
    for (key, score) in max_logits.items():
        output_scores[key] = (args.na_lambda * null_odds[key] + (1-args.na_lambda) * cls_score[key]) - max_logits[key]
                
    best_th = args.thresh

    for qid in output_predictions.keys():
        if output_scores[qid] > best_th:
            output_predictions[qid] = ""

    #Save files
    output_prediction_file = os.path.join(args.output_dir, "predictions.json")
    output_null_log_odds_file = os.path.join(args.output_dir, "combined_null_log_odds.json")
    with open(output_prediction_file, "w") as writer:
        writer.write(json.dumps(output_predictions, indent=4) + "\n")
    with open(output_null_log_odds_file, "w") as writer:
        writer.write(json.dumps(output_scores, indent=4) + "\n")
        
        
    result = eval_squad(args.predict_file, output_prediction_file, output_null_log_odds_file,
                            args.thresh)

    results = {}
    result = dict((k + ('_{}'.format(global_step) if global_step else ''), v) for k, v in result.items())
    results.update(result)
        
    logger.info("Results: {}".format(results))
    with open(os.path.join(args.output_dir, "result.txt"), "a") as writer:
        for key in sorted(results.keys()):
            logger.info("  %s = %s", key, str(results[key]))
            writer.write("%s = %s\t" % (key, str(results[key])))
            writer.write("\t\n")
    return results
        
        
def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument('--input_cls_files', type=str, default="cls_score.json,cls_av_score.json")
    parser.add_argument('--input_av_null', type=str, default="null_odds.json")
    parser.add_argument('--input_nbest_files', type=str, default="nbest_predictions.json")
    parser.add_argument('--output_dir', type=str, default="")
    parser.add_argument('--thresh', default=0, type=float)
    parser.add_argument("--predict_file", default="data/dev-v2.0.json")
    parser.add_argument('--na_lambda', type=float, default=0.5,
                        help="Lambda 1 for score_na. This is the weight of Null Odds (s1 + e1). CLS lambda will be 1 minus this value")
    parser.add_argument('--cls_beta', type=float, default=0.5,
                        help="Beta 1 for cls_scores (v). This is the weight of EFV (cls_score file). IFV weight will be 1 minus this value")
    args = parser.parse_args()
    get_score1(args)

if __name__ == "__main__":
    main()
