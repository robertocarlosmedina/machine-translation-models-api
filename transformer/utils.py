import os
import zipfile
import shutil
from termcolor import colored

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import torch


def train(model, iterator, optimizer, criterion, clip, epoch, progress_bar, **kwargs):
    
    model.train()

    epoch_loss = 0
    train_acc, correct_train, target_count = 0, 0, 0

    len_iterator = len(iterator)

    for i, batch in enumerate(iterator):

        src = batch.src
        trg = batch.trg

        optimizer.zero_grad()

        output, _ = model(src, trg[:, :-1])

        # output = [batch size, trg len - 1, output dim]
        # trg = [batch size, trg len]

        output_dim = output.shape[-1]

        output = output.contiguous().view(-1, output_dim)
        trg = trg[:, 1:].contiguous().view(-1)

        # output = [batch size * trg len - 1, output dim]
        # trg = [batch size * trg len - 1]

        loss = criterion(output, trg)

        _, predicted = torch.max(output.data, 1)
        target_count += trg.size(0)
        correct_train += (trg == predicted).sum().item()
        train_acc += (correct_train) / target_count

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

        optimizer.step()

        epoch_loss += loss.item()
        progress_bar.set_postfix(
            epoch=f" {epoch}, train loss= {round(epoch_loss / (i + 1), 4)}, train accu: {train_acc / (i + 1):.4f}", 
            refresh=True)
        progress_bar.update()

    print("\n")

    return epoch_loss / len_iterator, train_acc / len_iterator


def evaluate(model, iterator, criterion, epoch, progress_bar):

    model.eval()

    epoch_loss = 0
    train_acc, correct_train, target_count = 0, 0, 0

    with torch.no_grad():

        for i, batch in enumerate(iterator):

            src = batch.src
            trg = batch.trg

            output, _ = model(src, trg[:, :-1])

            # output = [batch size, trg len - 1, output dim]
            # trg = [batch size, trg len]

            output_dim = output.shape[-1]

            output = output.contiguous().view(-1, output_dim)
            trg = trg[:, 1:].contiguous().view(-1)

            # output = [batch size * trg len - 1, output dim]
            # trg = [batch size * trg len - 1]

            loss = criterion(output, trg)

            _, predicted = torch.max(output.data, 1)
            target_count += trg.size(0)
            correct_train += (trg == predicted).sum().item()
            train_acc += (correct_train) / target_count

            epoch_loss += loss.item()
            progress_bar.set_postfix(
                epoch=f" {epoch}, train loss= {round(epoch_loss / (i + 1), 4)}, train accu: {train_acc / (i + 1):.4f}", 
                refresh=True)
            progress_bar.update()
    
    print("\n")

    return epoch_loss / len(iterator), train_acc / len(iterator)
    

def translate_sentence(cv_nlp, sentence, src_field, trg_field, model, device, max_len = 50):
    
    model.eval()
        
    if isinstance(sentence, str):
        tokens = [token.text.lower() for token in cv_nlp(sentence)]
    else:
        tokens = [token.lower() for token in sentence]

    tokens = [src_field.init_token] + tokens + [src_field.eos_token]
        
    src_indexes = [src_field.vocab.stoi[token] for token in tokens]

    src_tensor = torch.LongTensor(src_indexes).unsqueeze(0).to(device)
    
    src_mask = model.make_src_mask(src_tensor)
    
    with torch.no_grad():
        enc_src = model.encoder(src_tensor, src_mask)

    trg_indexes = [trg_field.vocab.stoi[trg_field.init_token]]

    for i in range(max_len):

        trg_tensor = torch.LongTensor(trg_indexes).unsqueeze(0).to(device)

        trg_mask = model.make_trg_mask(trg_tensor)
        
        with torch.no_grad():
            output, attention = model.decoder(trg_tensor, enc_src, trg_mask, src_mask)
        
        pred_token = output.argmax(2)[:,-1].item()
        
        trg_indexes.append(pred_token)

        if pred_token == trg_field.vocab.stoi[trg_field.eos_token]:
            break
    
    trg_tokens = [trg_field.vocab.itos[i] for i in trg_indexes]
    
    return trg_tokens[1:], attention


def display_attention(cv_nlp, sentence, translation, attention, n_heads = 1, n_rows = 1, n_cols = 1):
    
    assert n_rows * n_cols == n_heads

    if isinstance(sentence, str):
        tokens = [token.text.lower() for token in cv_nlp(sentence)]
    else:
        tokens = [token.lower() for token in sentence]
    
    fig = plt.figure(figsize=(15,25))
    
    for i in range(n_heads):
        
        ax = fig.add_subplot(n_rows, n_cols, i+1)
        
        _attention = attention.squeeze(0)[i].cpu().detach().numpy()

        cax = ax.matshow(_attention, cmap='bone')

        ax.tick_params(labelsize=12)
        ax.set_xticklabels(['']+['<sos>']+[t.lower() for t in tokens]+['<eos>'], 
                           rotation=45)
        ax.set_yticklabels(['']+translation)

        ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
        ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    plt.show()
    plt.close()


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def save_checkpoint(state, filename="checkpoints/my_checkpoint.pth.tar"):
    print(colored("=> Saving checkpoint", 'cyan'))
    torch.save(state, filename)


def load_checkpoint(checkpoint, model, optimizer):
    print(colored("=> Loading checkpoint", "cyan"))
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])


def download_crioleSet() -> None:
    try:
        os.system("wget https://github.com/robertocarlosmedina/crioleSet/archive/main.zip")
        zip_object = zipfile.ZipFile(f"main.zip", "r")
        zip_object.extractall(".data")
        os.rename(".data/crioleSet-main", ".data/crioleSet")
        os.remove(".data/crioleSet/main.py")
        os.remove(".data/crioleSet/README.md")
        os.remove(".data/crioleSet/RULES USED.txt")
        shutil.rmtree(".data/crioleSet/src")
        os.remove("main.zip")

        print(
            colored("==> The crioleSet dataset has been added to the project", attrs=["bold"]))
    except:
        print(
            colored("==> Error downloading the crioleSet dataset", "red", attrs=["bold"]))


def check_dataset() -> None:
    if not os.path.isdir(".data"):
        download_crioleSet()
    else: 
        print(colored("==> The crioleSet is in the project", attrs=["bold"]))

def progress_bar(value: int, max_width: int, display: str, unit: str, bar_size=20):
    bar_state = int((bar_size*value)/max_width)
    print(
        colored(f" [{'='*bar_state}>{' '*(bar_size-bar_state)}] {value}/{max_width} {unit}, {display}", attrs=["bold"]), 
        end='\r'
    )
