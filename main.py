from trainer import Trainer


def main():
    trainer = Trainer()
    
    for epoch in range(0, 20):
        trainer.cur_epoch = epoch
        trainer.fit()
        trainer.save(epoch + 1)
        trainer.score('test')
        
#     trainer.load(20)
#     trainer.score('test')

    
if __name__ == "__main__":
    main()