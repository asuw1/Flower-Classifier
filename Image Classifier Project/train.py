import torch
from torch import nn, optim
from torchvision import datasets, transforms
from util import get_model, get_input_args_train

def train(model, trainloader, valloader, criterion, optimizer, device, epochs, print_every=10):
    for e in range(epochs):
        running_loss = 0
        model.train()
        for ii, (images, labels) in enumerate(trainloader):
            # Ensuring data is stored in the correct device
            images, labels = images.to(device), labels.to(device)
            
            # Zeroing Gradients
            optimizer.zero_grad()
    
            # Training steps
            outputs = model.forward(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
    
            # Evaluating every print_every steps
            if (ii+1) % print_every == 0:
                val_loss = 0
                accuracy = 0
                model.eval()
                with torch.no_grad():
                    for images, labels in valloader:
                        images, labels = images.to(device), labels.to(device)
                        out = model.forward(images)
                        batch_loss = criterion(out, labels)
                        val_loss += batch_loss.item()
                        
                        # Accuracy
                        ps = torch.exp(out)
                        top_p, top_class = ps.topk(1, dim=1)
                        equality = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equality.type(torch.FloatTensor)).item()
                
                print(f"Epoch {e+1}/{epochs}.."
                    f"Step {ii+1}/{len(trainloader)}.. "
                    f"Train loss: {running_loss/print_every:.3f}.. "
                    f"Validation loss: {val_loss/len(valloader):.3f}.. "
                    f"Validation accuracy: {accuracy/len(valloader):.3f}")
                
                # Enabling training mode again
                running_loss = 0
                model.train()

def save_checkpoint(model, path, arch, hidden_units, learning_rate, epochs, class_to_idx):
    checkpoint = {
        'arch': arch,
        'hidden_units': hidden_units,
        'state_dict': model.classifier.state_dict(),
        'classifier': model.classifier,
        'class_to_idx': class_to_idx,
        'learning_rate': learning_rate,
        'epochs': epochs
    }
    torch.save(checkpoint, path)
        
def main():
    # Get arguments
    args = get_input_args_train()

    # Transform data
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                           transforms.RandomResizedCrop(224),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])])
    
    
    valtest_transforms = transforms.Compose([transforms.Resize(255),
                                             transforms.CenterCrop(224),
                                             transforms.ToTensor(),
                                             transforms.Normalize([0.485, 0.456, 0.406],
                                                                  [0.229, 0.224, 0.225])])
    
    train_dataset = datasets.ImageFolder(args.data_dir + '/train', transform=train_transforms)
    valid_dataset = datasets.ImageFolder(args.data_dir + '/valid', transform=valtest_transforms)
    
    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
    valloader = torch.utils.data.DataLoader(valid_dataset, batch_size=64)
    
    # Get model
    model = get_model(args.arch, args.hidden_units)
    
    # Define loss and optimizer
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=args.learning_rate)
    
    # Use GPU if specified
    device = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Train the model
    train(model, trainloader, valloader, criterion, optimizer, device, args.epochs)
    
    # Save the checkpoint
    save_checkpoint(model, args.save_dir + '/checkpoint.pth', args.arch, args.hidden_units, args.learning_rate, args.epochs, \
                    train_dataset.class_to_idx)

if __name__ == '__main__':
    main()