import yaml
import click

from train import *
from predict import *
from reclass import *

def execute(configPath, doTrain, doPrediction, doReclass):

    with open(configPath, "r") as config:
        params = yaml.safe_load(config)["Cluster"]

    # train
    if doTrain:
        train(params)
    else:
        pass
    # predict
    if doPrediction:
        if doTrain:
            params.update({
                "dir_model": os.path.join(params['dir_train'], "model.sav")
            })
        else:
            pass
        predict(params)
    else:
        pass
    # reclass can be executed independently
    if doReclass:
        reclass(params)
    else:
        pass




@click.command()
@click.option('--config', default='../config.yaml',
              help='Directory of the config file')
@click.option('--do-train', is_flag=True, help='Do train model')
@click.option('--do-predict', is_flag=True, help='Do prediction on specified data')
@click.option('--do-reclass', is_flag=True, help='Reclass predicted data from kmean model')
def main(config, do_train, do_predict, do_reclass):
    execute(config, do_train, do_predict, do_reclass)

if __name__=='__main__':
    main()


