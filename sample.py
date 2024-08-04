import warnings

# from collections import defaultdict
from typing import Dict, List, Tuple

import pandas as pd
import torch
from torch.nn.utils import clip_grad_norm_

# from torch.utils.data import DataLoader
from tqdm import tqdm

from .data_processor import DataProcessor
from .general import corrcoef_loss, get_strategy_returns, share_loss
from .module import Multi_Task_Model  # CustomDataset
from .utils import all_nan, plot_pred_nan_num


class MTL_TSMOM:
    def __init__(
        self,
        dataset: DataProcessor,
        input_size: int,
        lstm_hidden_size: int,
        mlp_hidden_size: int,
        lstm_layers: int,
        mlp_layers: int,
        optimizer_name: str,
        transcation_cost: float,
        target_vol: float,
        lstm_dropout: float,
        mlp_dropout: float,
        max_grad_norm: float,
        # batch_size: int,
        num_epochs: int,
        opt_kwargs: Dict = None,
        early_stopping: int = 50,
        log_step: int = 100,
        verbose: bool = False,
        save_path: str = None,
    ) -> None:
        self.epoch_loss = []  # 储存每一次的损失
        # self.all_loss = defaultdict(list)  # 储存每一次的损失 1-train 2-valid
        self.dataset = dataset
        self.transcation_cost = transcation_cost
        self.target_vol = target_vol
        self.max_grad_norm = max_grad_norm
        # self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.early_stopping = early_stopping
        self.log_step = log_step
        self.verbose = verbose
        self.save_path = save_path

        # 初始化模型
        self.model = Multi_Task_Model(
            input_size,
            lstm_hidden_size,
            mlp_hidden_size,
            lstm_layers,
            mlp_layers,
            lstm_dropout,
            mlp_dropout,
        ).cuda()

        if opt_kwargs is None:
            opt_kwargs = {}

        self.optimizer = getattr(torch.optim, optimizer_name)(
            self.model.parameters(), **opt_kwargs
        )

    def log(self, arg, verbose=True) -> None:
        if verbose:
            print(arg)

    def train_model(self, train_datase: List, gloabal_step: int = None) -> float:
        self.model.train()
        # train_dataset = CustomDataset(train_datase)
        # train_loader = DataLoader(
        #     train_dataset, batch_size=self.batch_size, shuffle=False
        # )
        features, next_returns, forward_vol = train_datase
        total_loss = 0.0
        # loss = 0.0
        # for batch, (features, next_returns, forward_vol) in enumerate(train_loader):
        pred_sigma, weight = self.model(features)
        auxiliary_loss: float = corrcoef_loss(pred_sigma, forward_vol)
        main_loss: float = share_loss(
            weight, next_returns, self.target_vol, self.transcation_cost
        )
        total_loss = (auxiliary_loss + main_loss) * 0.5
        self.optimizer.zero_grad()
        total_loss.backward()
        # 为了防止梯度爆炸，我们对梯度进行裁剪
        if self.max_grad_norm is not None:
            clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

        self.optimizer.step()
        # if gloabal_step is not None:
        #     self.all_loss[gloabal_step].append(
        #         (1, batch, auxiliary_loss, main_loss, total_loss)
        #     )
        # loss += total_loss
        return total_loss  # loss / len(train_loader)

    def validation_model(
        self, validation_dataset: List, gloabal_step: int = None
    ) -> float:
        # valid_dataset = CustomDataset(validation_dataset)
        # valid_loader = DataLoader(
        #     valid_dataset, batch_size=self.batch_size, shuffle=False
        # )
        total_loss = 0.0

        # loss = 0.0
        self.model.eval()
        features, next_returns, forward_vol = validation_dataset
        with torch.no_grad():
            # for batch, (features, next_returns, forward_vol) in enumerate(valid_loader):
            pred_sigma, weight = self.model(features)

            auxiliary_loss = corrcoef_loss(pred_sigma, forward_vol)
            main_loss = share_loss(
                weight, next_returns, self.target_vol, self.transcation_cost
            )

            total_loss = (auxiliary_loss + main_loss) * 0.5
            # loss += total_loss
            # if gloabal_step is not None:
            #     self.all_loss[gloabal_step].append(
            #         (2, batch, auxiliary_loss, main_loss, total_loss)
            #     )
        return total_loss  # loss / len(valid_loader)

    def predict_data(self, test_part: List) -> Tuple[torch.Tensor, torch.Tensor]:
        features, next_returns, _ = test_part
        with torch.no_grad():
            _, weight = self.model(features)
        return weight, next_returns

    def loop(
        self, train_part: List, valid_part: List, global_step: int = None
    ) -> float:
        best_valid_loss: float = float("inf")  # 用于记录最好的验证集损失
        epochs_without_improvement: int = 0  # 用于记录连续验证集损失没有改善的轮数
        for epoch in range(self.num_epochs):
            train_loss: float = self.train_model(train_part)
            valid_loss: float = self.validation_model(valid_part)

            if (self.log_step is not None) and (epoch % self.log_step == 0):
                self.log(
                    f"Epoch {epoch or epoch+1}, Train Loss: {train_loss:.4f}, Valid Loss: {valid_loss:.4f}",
                    self.verbose,
                )

            # 判断是否有性能提升，如果没有则计数器加 1
            # NOTE:这样是最小化适用的,如果是最大化,需要改成 valid_loss > best_valid_loss
            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                epochs_without_improvement: int = 0
            else:
                epochs_without_improvement += 1

            # 保存每一次的损失
            self.epoch_loss.append((global_step, train_loss, valid_loss))
            # 判断是否满足 early stopping 条件
            if (self.early_stopping is not None) and (
                epochs_without_improvement >= self.early_stopping
            ):
                self.log(f"Early stopping at epoch {epoch + 1}...", self.verbose)
                break

        return valid_loss

    def fit(self):
        ls: List = [] # 储存每一次的权重和收益
        size: int = len(self.dataset.train_dataset)
        for i, (train_part, valid_part, test_part) in enumerate(
            tqdm(
                zip(
                    self.dataset.train_dataset,
                    self.dataset.valid_dataset,
                    self.dataset.test_dataset,
                ),
                total=size,
                desc="train",
            )
        ):
            self.loop(train_part, valid_part, i)
            weight, next_returns = self.predict_data(test_part)
            ls.append((weight, next_returns))
            if all_nan(weight):
                warnings.warn(f"下标{i}次时:All nan in weight,已经跳过")
                # raise ValueError(f"下标{i}次时:All nan in weight")
                break

        weights_tensor: torch.Tensor = torch.cat([t[0] for t in ls], dim=0)
        returns_tensor: torch.Tensor = torch.cat([t[1] for t in ls], dim=0)

        self.weight = weights_tensor
        self.next_returns = returns_tensor
        if self.save_path is not None:
            torch.save(self.model.state_dict(), self.save_path)
        # return weights_tensor, returns_tensor

    def get_backtest_returns(self) -> pd.DataFrame:
        try:
            self.weight
        except NameError as e:
            raise NameError("请先调用fit方法") from e
        strategy_frame: pd.DataFrame = get_strategy_returns(
            self.weight, self.next_returns, self.dataset.test_idx
        )
        return strategy_frame

    def get_loss_score(self) -> pd.DataFrame:
        if self.epoch_loss == []:
            raise ValueError("请先调用fit方法")
        return pd.DataFrame(
            [(j.item(), k.item()) for _, j, k in self.epoch_loss],
            columns=["train", "valid"],
        )

    def plot_pred_nan_num(self):
        try:
            self.weight
        except NameError as e:
            raise NameError("请先调用fit方法") from e
        return plot_pred_nan_num(self.weight)
