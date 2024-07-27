import torch
import json
from util import load_model, process_image, get_input_args_pred

def predict(img_path, model, topk, device):
    img = process_image(img_path)
    img = torch.from_numpy(img).type(torch.FloatTensor)
    img = img[None, :, :, :]
    model.to(device)
    img = img.to(device)

    with torch.no_grad():
        model.eval()
        output = model.forward(img)
    probs = torch.exp(output)
    topk_probs, topk_indices = probs.topk(topk, dim=1)
    topk_probs, topk_indices = topk_probs.cpu(), topk_indices.cpu()
    topk_probs, topk_indices = topk_probs.tolist(), topk_indices.tolist()
    
    idx_to_class = model.class_to_idx
    idx_to_class = {val: key for key, val in idx_to_class.items()}
    cls = [idx_to_class[x] for x in topk_indices[0]]

    return (topk_probs[0], cls)

def main():
    args = get_input_args_pred()

    device = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")
    model = load_model(args.checkpoint_path)

    probs, cls = predict(args.file_path, model, args.top_k, device)

    print('Probabilities\tClasses')
    
    if args.category_names:
        cat_to_name = None
        with open(args.category_names, 'r') as f:
            cat_to_name = json.load(f)
        cls = [cat_to_name[cl] for cl in cls]
    
    for ii in range(args.top_k):
        print(f"{probs[ii]:.4f} \t {cls[ii]}")

if __name__ == "__main__":
    main()