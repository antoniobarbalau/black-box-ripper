import argparse
import setup
import trainer

if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser(description = 'Experiment setup')

    arg_parser.add_argument('--epochs', type = str, default = '200')
    arg_parser.add_argument('--generator', type = str, default = 'gan')
    arg_parser.add_argument('--optim', type = str, default = 'adam')
    arg_parser.add_argument('--proxy_dataset', type = str, default = 'cifar10')
    arg_parser.add_argument('--sample_optimization', type = str, default = 'class')
    arg_parser.add_argument('--samples', type = str, default = 'optimized')
    arg_parser.add_argument('--size', type = int, default = 32)
    arg_parser.add_argument('--student', type = str, default = 'half_lenet')
    arg_parser.add_argument('--teacher', type = str, default = 'lenet')
    arg_parser.add_argument('--true_dataset', type = str, default = 'split_fmnist')

    env = arg_parser.parse_args()

    teacher, teacher_dataset, student = setup.prepare_teacher_student(env)
    trainer.evaluate(teacher, teacher_dataset)
    generator = setup.prepare_generator(env)

    student_dataset = setup.prepare_student_dataset(
        env, teacher, teacher_dataset, student, generator
    )

    if env.optim == 'sgd':
        trainer.train_or_restore_predictor(
            student, student_dataset, loss_type = 'binary',
            n_epochs = int(env.epochs)
        )
    else:
        trainer.train_or_restore_predictor_adam(
            student, student_dataset, loss_type = 'binary',
            n_epochs = int(env.epochs)
        )
    trainer.evaluate(student, teacher_dataset)


