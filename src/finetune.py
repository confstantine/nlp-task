import torch
import time
from tqdm import tqdm
import sklearn.metrics as sklearn_metrics
from torch.utils.data import DataLoader
from torch.utils.data._utils.collate import default_collate
from utils import get_optimizer, get_loss
from ema import EMA


class Task(object):
    def __init__(
            self,
            module,
            optimizer,
            loss_function,
            class_num=None,
            scheduler=None,
            n_gpu=1,
            device=None,
            cuda_device=0,
            ema_decay=None,
            **kwargs
    ):

        self.module = module
        self.optimizer = optimizer
        self.loss_function = get_loss(loss_function)

        self.class_num = class_num
        self.scheduler = scheduler

        self.n_gpu = n_gpu

        if device == None:
            if torch.cuda.is_available():
                if cuda_device == -1:
                    self.device = torch.device("cuda")
                else:
                    self.device = torch.device(f"cuda:{cuda_device}")
            else:
                self.device = "cpu"

        self.module.to(self.device)

        if self.n_gpu > 1:
            self.module = torch.nn.DataParallel(self.module)

        self.ema_decay = ema_decay
        if self.ema_decay:
            self.ema = EMA(self.module.parameters(), decay=self.ema_decay)

    def _collate_fn(self, batch):
        return default_collate(batch)

    def _prepare_train_begin(self, **kwargs):
        pass

    def _on_train_begin(self, **kwargs):
        pass

    def _finish_train_begin(self, **kwargs):
        pass

    def _prepare_train_begin_record(self, **kwargs):
        pass

    def _on_train_begin_record(self, **kwargs):
        pass

    def _finish_train_begin_record(self, **kwargs):
        pass

    def _prepare_epoch_begin(self, **kwargs):
        pass

    def _on_epoch_begin(self, **kwargs):
        pass

    def _finish_epoch_begin(self, **kwargs):
        pass

    def _prepare_epoch_begin_record(self, **kwargs):
        pass

    def _on_epoch_begin_record(self, **kwargs):
        pass

    def _finish_epoch_begin_record(self, **kwargs):
        pass

    def _prepare_step_begin(self, **kwargs):
        pass

    def _on_step_begin(self, **kwargs):
        pass

    def _finish_step_begin(self, **kwargs):
        pass

    def _prepare_step_begin_record(self, **kwargs):
        pass

    def _on_step_begin_record(self, **kwargs):
        pass

    def _finish_step_begin_record(self, **kwargs):
        pass

    def _prepare_compute_loss(self, **kwargs):
        pass

    def _compute_loss(self, **kwargs):
        pass

    def _finish_compute_loss(self, **kwargs):
        pass

    def _prepare_compute_loss_record(self, **kwargs):
        pass

    def _compute_loss_record(self, **kwargs):
        pass

    def _finish_compute_loss_record(self, **kwargs):
        pass

    def _prepare_backward(self, **kwargs):
        pass

    def _on_backward(self, ):
        pass

    def _finish_backward(self, **kwargs):
        pass

    def _prepare_backward_record(self, **kwargs):
        pass

    def _on_backward_record(self, **kwargs):
        pass

    def _finish_backward_record(self, **kwargs):
        pass

    def _prepare_optimize(self, **kwargs):
        pass

    def _on_optimize(self, **kwargs):
        pass

    def _finish_optimize(self, **kwargs):
        pass

    def _prepare_optimize_record(self, **kwargs):
        pass

    def _on_optimize_record(self, **kwargs):
        pass

    def _finish_optimize_record(self, **kwargs):
        pass

    def _prepare_step_end(self, **kwargs):
        pass

    def _on_step_end(self, **kwargs):
        pass

    def _finish_step_end(self, **kwargs):
        pass

    def _prepare_step_end_record(self, **kwargs):
        pass

    def _on_step_end_record(self, **kwargs):
        pass

    def _finish_step_end_record(self, **kwargs):
        pass

    def _prepare_epoch_end(self, **kwargs):
        pass

    def _on_epoch_end(self, **kwargs):
        pass

    def _finish_epoch_end(self, **kwargs):
        pass

    def _prepare_epoch_end_record(self, **kwargs):
        pass

    def _on_epoch_end_record(self, **kwargs):
        pass

    def _finish_epoch_end_record(self, **kwargs):
        pass

    def _prepare_train_end(self, **kwargs):
        pass

    def _on_train_end(self, **kwargs):
        pass

    def _finish_train_end(self, **kwargs):
        pass

    def _prepare_train_end_record(self, **kwargs):
        pass

    def _on_train_end_record(self, **kwargs):
        pass

    def _finish_train_end_record(self, **kwargs):
        pass

    def _prepare_fit(self, **kwargs):
        pass

    def fit(self, **kwargs):
        pass

    def _finish_fit(self, **kwargs):
        pass

    def _prepare_evaluate(self, **kwargs):
        pass

    def evaluate(self, **kwargs):
        pass

    def _finish_evaluate(self, **kwargs):
        pass

    def _on_evaluate_epoch_begin_record(self, **kwargs):
        pass

    def _on_evaluate_epoch_end(self, **kwargs):
        pass

    def _on_evaluate_epoch_end_record(self, **kwargs):
        pass

    def _on_evaluate_end(self, **kwargs):
        pass

    def _on_evaluate_end_record(self, **kwargs):
        pass

    def _get_module_inputs_on_train(self):
        pass

    def _get_module_label_on_train(self):
        pass

    def _get_module_inputs_on_eval(self):
        pass

    def _get_module_label_on_eval(self):
        pass


class SequenceClassificationTask(Task):

    def __init__(self, *args, **kwargs):

        super(SequenceClassificationTask, self).__init__(*args, **kwargs)
        if hasattr(self.module, 'task') is False:
            self.module.task = 'SequenceLevel'

    def _on_train_begin(
            self,
            train_data,
            validation_data,
            batch_size,
            lr,
            params,
            shuffle,
            train_to_device_cols=None,
            **kwargs
    ):

        if self.class_num == None:
            self.class_num = train_data.class_num

        if train_to_device_cols == None:
            self.train_to_device_cols = train_data.to_device_cols
        else:
            self.train_to_device_cols = train_to_device_cols

        train_generator = DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn=self._collate_fn)
        self.train_generator_lenth = len(train_generator)

        self.optimizer = get_optimizer(self.optimizer, self.module, lr, params)
        self.optimizer.zero_grad()

        self.module.train()

        self._on_train_begin_record(**kwargs)

        return train_generator

    def _on_train_begin_record(self, **kwargs):

        self.logs['global_step'] = 0
        self.logs['tr_loss'] = 0
        self.logs['logging_loss'] = 0

        return self.logs

    def _on_epoch_begin(self, **kwargs):

        self.module.train()

        self._on_epoch_begin_record(**kwargs)

    def _on_epoch_begin_record(self, **kwargs):

        self.logs['epoch_loss'] = 0
        self.logs['epoch_evaluation'] = 0
        self.logs['epoch_step'] = 0

    def _on_step_begin(
            self,
            epoch,
            step,
            inputs,
            **kwargs
    ):

        self._on_step_begin_record(epoch, step, inputs, **kwargs)

    def _on_step_begin_record(
            self,
            epoch,
            step,
            inputs,
            **kwargs
    ):
        pass

    def _get_train_loss(
            self,
            inputs,
            logits,
            verbose=True,
            **kwargs
    ):
        # 计算损失
        loss = self._compute_loss(inputs, logits, **kwargs)

        self._compute_loss_record(inputs, logits, loss, verbose, **kwargs)

        return loss

    def _compute_loss(
            self,
            inputs,
            logits,
            verbose=True,
            **kwargs
    ):
        loss = self.loss_function(logits, inputs['label_ids'])

        return loss

    def _compute_loss_record(
            self,
            inputs,
            logits,
            loss,
            verbose,
            **kwargs
    ):

        self.logs['epoch_loss'] += loss.item()
        self.logs['epoch_step'] += 1
        self.logs['global_step'] += 1

    def _on_backward(
            self,
            inputs,
            logits,
            loss,
            gradient_accumulation_steps=1,
            grad_clip=None,
            **kwargs
    ):

        # 如果GPU数量大于1
        if self.n_gpu > 1:
            loss = loss.mean()
        # 如果使用了梯度累积，除以累积的轮数
        if gradient_accumulation_steps > 1:
            loss = loss / gradient_accumulation_steps

        loss.backward()

        if grad_clip != None:
            torch.nn.utils.clip_grad_norm_(self.module.parameters(), grad_clip)

        self._on_backward_record(**kwargs)

        return loss

    def _on_optimize(
            self,
            step,
            gradient_accumulation_steps=1,
            **kwargs
    ):

        if (step + 1) % gradient_accumulation_steps == 0:
            self.optimizer.step()  # 更新权值

            if self.ema_decay:
                self.ema.update(self.module.parameters())

            if self.scheduler:
                self.scheduler.step()  # 更新学习率

            self.optimizer.zero_grad()  # 清空梯度

            self._on_optimize_record(**kwargs)

        return step

    def _on_step_end(
            self,
            step,
            inputs,
            logits,
            loss,
            verbose=True,
            show_step=100,
            **kwargs
    ):

        if verbose and (step + 1) % show_step == 0:
            print('[{}/{}],train loss is:{:.6f}'.format(
                step,
                self.train_generator_lenth,
                self.logs['epoch_loss'] / self.logs['epoch_step']))

        self._on_step_end_record(**kwargs)

    def _on_epoch_end(
            self,
            epoch,
            verbose=True,
            **kwargs
    ):

        if verbose:
            print('epoch:[{}],train loss is:{:.6f} \n'.format(
                epoch,
                self.logs['epoch_loss'] / self.logs['epoch_step']))

    def _on_evaluate_begin(
            self,
            validation_data,
            batch_size,
            shuffle,
            evaluate_to_device_cols=None,
            **kwargs
    ):
        if evaluate_to_device_cols == None:
            self.evaluate_to_device_cols = validation_data.to_device_cols
        else:
            self.evaluate_to_device_cols = evaluate_to_device_cols

        generator = DataLoader(validation_data, batch_size=batch_size, shuffle=False, collate_fn=self._collate_fn)

        self.module.eval()

        self._on_evaluate_begin_record(**kwargs)

        return generator

    def _on_evaluate_begin_record(self, **kwargs):

        self.evaluate_logs['eval_loss'] = 0
        self.evaluate_logs['eval_acc'] = 0
        self.evaluate_logs['eval_step'] = 0
        self.evaluate_logs['eval_example'] = 0

        self.evaluate_logs['labels'] = []
        self.evaluate_logs['logits'] = []

    def _on_evaluate_epoch_begin(self, **kwargs):

        if self.ema_decay:
            self.ema.store(self.module.parameters())
            self.ema.copy_to(self.module.parameters())

        self._on_evaluate_epoch_begin_record(**kwargs)

    def _get_evaluate_loss(
            self,
            inputs,
            logits,
            verbose=True,
            **kwargs
    ):
        # 计算损失
        loss = self._compute_loss(inputs, logits, **kwargs)

        return loss

    def _on_evaluate_step_end(self, inputs, logits, **kwargs):

        with torch.no_grad():
            # compute loss
            loss = self._get_evaluate_loss(inputs, logits, **kwargs)

            labels = inputs['label_ids'].cpu()
            logits = logits.cpu()

            _, preds = torch.max(logits, 1)

        self.evaluate_logs['labels'].append(labels)
        self.evaluate_logs['logits'].append(logits)

        self.evaluate_logs['eval_example'] += len(labels)
        self.evaluate_logs['eval_step'] += 1
        self.evaluate_logs['eval_loss'] += loss.item()
        self.evaluate_logs['eval_acc'] += torch.sum(preds == labels.data).item()

    def _on_evaluate_epoch_end(
            self,
            validation_data,
            epoch=1,
            is_evaluate_print=True,
            **kwargs
    ):

        labels_ = torch.cat(self.evaluate_logs['labels'], dim=0)
        preds_ = torch.argmax(torch.cat(self.evaluate_logs['logits'], dim=0), -1)

        f1_score = sklearn_metrics.f1_score(labels_, preds_, average='macro')

        report_ = sklearn_metrics.classification_report(labels_,
                                                        preds_,
                                                        target_names=[str(category_) for category_ in
                                                                      validation_data.categories])

        confusion_matrix_ = sklearn_metrics.confusion_matrix(labels_, preds_)

        if is_evaluate_print:
            print('classification_report: \n', report_)
            print('confusion_matrix_: \n', confusion_matrix_)
            print('test loss is:{:.6f},test acc is:{:.6f},f1_score is:{:.6f}'.format(
                self.evaluate_logs['eval_loss'] / self.evaluate_logs['eval_step'],
                self.evaluate_logs['eval_acc'] / self.evaluate_logs['eval_example'],
                f1_score))

    def _on_evaluate_end(
            self,
            evaluate_save=False,
            save_module_path=None,
            **kwargs
    ):

        if evaluate_save:
            if save_module_path is None:
                prefix = './checkpoint/' + str(self.module.__class__.__name__) + '_'
                save_module_path = time.strftime(prefix + '%m%d_%H:%M:%S.pth')

            torch.save(self.module.state_dict(), save_module_path)

        self._on_evaluate_end_record()

        if self.ema_decay:
            self.ema.restore(self.module.parameters())

    def _get_module_inputs_on_train(
            self,
            inputs,
            **kwargs
    ):
        for col in self.train_to_device_cols:
            inputs[col] = inputs[col].to(self.device)

        return inputs

    def _get_module_inputs_on_eval(
            self,
            inputs,
            **kwargs
    ):
        for col in self.evaluate_to_device_cols:
            inputs[col] = inputs[col].to(self.device)

        return inputs

    def fit(
            self,
            train_data=None,
            validation_data=None,
            lr=False,
            params=None,
            batch_size=32,
            epochs=1,
            **kwargs
    ):
        self.logs = dict()

        self.id2cat = train_data.id2cat

        train_generator = self._on_train_begin(train_data, validation_data, batch_size, lr, params, shuffle=True,
                                               **kwargs)

        for epoch in range(epochs):

            self._on_epoch_begin(**kwargs)

            for step, inputs in enumerate(tqdm(train_generator)):
                self._on_step_begin(epoch, step, inputs, **kwargs)

                inputs = self._get_module_inputs_on_train(inputs, **kwargs)

                # forward
                logits = self.module(**inputs)

                # 计算损失
                loss = self._get_train_loss(inputs, logits, **kwargs)

                # loss backword
                loss = self._on_backward(inputs, logits, loss, **kwargs)

                # optimize
                step = self._on_optimize(step, **kwargs)

                # setp evaluate
                self._on_step_end(step, inputs, logits, loss, **kwargs)

            self._on_epoch_end(epoch, **kwargs)

            if validation_data is not None:
                self.evaluate(validation_data, **kwargs)

        self._on_train_end(**kwargs)

    def evaluate(
            self,
            validation_data,
            evaluate_batch_size=16,
            return_pred=False,
            **kwargs
    ):
        self.evaluate_logs = dict()

        generator = self._on_evaluate_begin(validation_data, evaluate_batch_size, shuffle=False, **kwargs)

        with torch.no_grad():
            self._on_evaluate_epoch_begin(**kwargs)

            for step, inputs in enumerate(generator):
                inputs = self._get_module_inputs_on_eval(inputs, **kwargs)

                # forward
                logits = self.module(**inputs)

                self._on_evaluate_step_end(inputs, logits, **kwargs)

            self._on_evaluate_epoch_end(validation_data, **kwargs)

        self._on_evaluate_end(**kwargs)