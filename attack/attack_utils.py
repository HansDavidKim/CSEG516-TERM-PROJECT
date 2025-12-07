import math
import torch
import torch.nn.functional as F


def compute_reward(state_output, action_output, target_class, 
                   w1=2.0, w2=2.0, w3=8.0, epsilon=1e-7):
    """
    Compute reward based on classifier outputs only.
    
    Args:
        state_output: Classifier logits for state image (1, num_classes)
        action_output: Classifier logits for action image (1, num_classes)
        target_class: Target class index
        w1, w2, w3: Reward weights for classifier scores
        epsilon: Small constant for numerical stability
        
    Returns:
        reward: Scalar reward value
    """
    y = torch.tensor([target_class], device=state_output.device)
    
    # Original formula from attack.py lines 34-37
    # score1 = log(softmax(state_output)[target])
    score1 = float(torch.mean(torch.diag(torch.index_select(
        torch.log(F.softmax(state_output, dim=-1)).data, 1, y))))
    
    # score2 = log(softmax(action_output)[target])
    score2 = float(torch.mean(torch.diag(torch.index_select(
        torch.log(F.softmax(action_output, dim=-1)).data, 1, y))))
    
    # score3 = log(max(epsilon, softmax(state)[target] - max(other_classes)))
    state_probs = F.softmax(state_output, dim=-1)
    target_prob = float(torch.index_select(state_probs.data, 1, y))
    
    # Get max probability from other classes
    other_probs = torch.cat((state_probs[0, :target_class], 
                             state_probs[0, target_class+1:]), dim=-1)
    max_other_prob = float(torch.max(other_probs, dim=-1)[0])
    
    score3 = math.log(max(epsilon, target_prob - max_other_prob))
    
    # reward = w1 * score1 + w2 * score2 + w3 * score3
    reward = w1 * score1 + w2 * score2 + w3 * score3
    
    return reward




def evaluate_confidence(classifier_output, target_class):
    """
    Evaluate confidence score for target class.
    
    Args:
        classifier_output: Classifier logits (1, num_classes)
        target_class: Target class index
        
    Returns:
        confidence: Confidence score (probability) for target class
    """
    y = torch.tensor([target_class], device=classifier_output.device)
    probs = F.softmax(classifier_output, dim=-1)
    confidence = float(torch.mean(torch.diag(torch.index_select(probs.data, 1, y))))
    return confidence
