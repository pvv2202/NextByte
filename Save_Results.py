import csv
from pathlib import Path

def save_bleu(results, model_name):
    
    file_path = Path('results/' + model_name + '_bleu.csv')
    
    with open(file_path, 'w') as f:
        writer = csv.writer(f)
        # results are in the 2-d list format compatable with csv writer
        writer.writerows(results)


def save_results(results, model_mode):
    with open(model_mode + '_results.txt', 'w') as f:
        # Write training results
        f.write("Training Results:\n")
        for epoch, (acc, loss, f1) in enumerate(zip(results['train_acc'], results['train_loss'], results['train_f1']), 1):
            f.write(f"Epoch {epoch}: Accuracy={acc['accuracy']}, Loss={loss}, F1={f1}\n")
        
        # Write evaluation results
        f.write("\nEvaluation Results:\n")
        for epoch, (acc, loss, f1) in enumerate(zip(results['eval_acc'], results['eval_loss'], results['eval_f1']), 1):
            f.write(f"Epoch {epoch}: Accuracy={acc['accuracy']}, Loss={loss}, F1={f1}\n")
        
        # Write test results
        f.write("\nTest Results:\n")
        f.write(f"Accuracy={results['test_acc']['accuracy']}, Loss={results['test_loss']}, F1={results['test_f1']}\n")
        
        
        