def save_training_report(train_config, model_info, training_results, data_config):
    """
    Save the training report to a CSV file.
    Args:
        train_config (dict): Configuration for training.
        model_info (dict): Information about the model.
        training_results (dict): Results from the training process.
        data_config (dict): Configuration for the dataset.
    """
    import csv
    import os
    from datetime import datetime

    model_name = train_config.get('Model', '')
        
    reports_dir = os.path.join(train_config['save_dir'], 'reports')
    if not os.path.exists(reports_dir):
        os.makedirs(reports_dir)
        
    output_path = os.path.join(reports_dir, f"reports.csv")
    
    row_data = {
        "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        
        "model_name": model_name,
        "flops": model_info.get('flops', '').replace(',', ''),
        "params": model_info.get('params', '').replace(',', ''),
        
        "alpha": train_config.get('alpha', ''),
        "beta": train_config.get('beta', ''),
        "gamma": train_config.get('gamma', ''),
        "t_zero": train_config.get('t', ''),
        "p": train_config.get('p', ''),
        "q": train_config.get('q', ''),
        "chnn": train_config.get('chnn', ''),
        "num_layers": train_config.get('num_layers', ''),
        
        "optimizer": train_config.get('optim', ''),
        "learning_rate": train_config.get('learning_rate', ''),
        "batch_size": data_config.get('batch_size', ''),
        "decoder_batch_size": train_config.get('dnc_batch', ''),
        "lr_scheduler": train_config.get('lrsch', ''),
        "weight_decay": train_config.get('weight_decay', ''),
        "weight_decay_tfmode": train_config.get('wd_tfmode', ''),
        "loss": train_config.get('loss', 'CrossEntropy'),
        
        "epochs": training_results.get('epochs', train_config.get('epochs', '')),
        "max_iou": training_results.get('max_iou', ''),
        "best_model_path": training_results.get('best_model_path', ''),
        
        "input_shape": str(data_config.get("input_shape", [3, data_config.get("w", ""), data_config.get("h", "")])),
        "dataset": data_config.get('dataset_name', ''),
        "edge": data_config.get('Edge', ''),
        "augmentation": data_config.get('Aug_dataset', '')
    }
    
    file_exists = os.path.isfile(output_path)

    headers = list(row_data.keys())
    
    with open(output_path, mode='a', newline='') as output_file:
        writer = csv.DictWriter(output_file, fieldnames=headers)

        if not file_exists:
            writer.writeheader()
        
        writer.writerow(row_data)

    print(f"Training report saved to {output_path}")
    return output_path