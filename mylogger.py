import torch
from loguru import logger

class Logger(object):
    def __init__(self, runs, info=None):
        self.info = info
        self.results = [[] for _ in range(runs)]
        self.runs = runs

    def add_result(self, run, result):
        assert len(result) == 3
        assert run >= 0 and run < len(self.results)
        self.results[run].append(result)

    def flush(self):
        del(self.results)
        self.results = [[] for _ in range(self.runs)]
        
    def print_statistics(self, ratio,  run=0):
        if run is not None:
            result = 100 * torch.tensor(self.results[run])
            argmax = result[:, 1].argmax().item()
            logger.info(f'Ratio {ratio :.4f}:')
            logger.info(f'Highest Train: {result[:, 0].max():.2f}')
            logger.info(f'Highest Valid: {result[:, 1].max():.2f}')
            logger.info(f'  Final Train: {result[argmax, 0]:.2f}')
            logger.info(f'   Final Test: {result[argmax, 2]:.2f}')
        else:
            result = 100 * torch.tensor(self.results)

            best_results = []
            for r in result:
                train1 = r[:, 0].max().item()
                valid = r[:, 1].max().item()
                train2 = r[r[:, 1].argmax(), 0].item()
                test = r[r[:, 1].argmax(), 2].item()
                best_results.append((train1, valid, train2, test))

            best_result = torch.tensor(best_results)

            logger.info(f'All runs:')
            r = best_result[:, 0]
            logger.info(f'Highest Train: {r.mean():.2f} ± {r.std():.2f}')
            r = best_result[:, 1]
            logger.info(f'Highest Valid: {r.mean():.2f} ± {r.std():.2f}')
            r = best_result[:, 2]
            logger.info(f'  Final Train: {r.mean():.2f} ± {r.std():.2f}')
            r = best_result[:, 3]
            logger.info(f'   Final Test: {r.mean():.2f} ± {r.std():.2f}')