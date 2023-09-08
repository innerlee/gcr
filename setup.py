from setuptools import find_packages, setup


def readme():
    with open('README.md', encoding='utf-8') as f:
        content = f.read()
    return content


if __name__ == '__main__':
    setup(
        name='gcr',
        version='0.0.1',
        description='Official code for "Get the Best of Both Worlds: '
        'Improving Accuracy and Transferability by '
        'Grassmann Class Representation (ICCV 2023)"',
        long_description=readme(),
        long_description_content_type='text/markdown',
        keywords='computer vision, image classification, '
        'geometric deep learning, Grassmann class representation',
        packages=find_packages(exclude=('configs', 'tools')),
        include_package_data=True,
        python_requires='>=3.7',
        classifiers=[
            'Development Status :: 4 - Beta',
            'License :: OSI Approved :: Apache Software License',
            'Operating System :: OS Independent',
            'Programming Language :: Python :: 3',
            'Programming Language :: Python :: 3.7',
            'Programming Language :: Python :: 3.8',
            'Programming Language :: Python :: 3.9',
            'Programming Language :: Python :: 3.10',
            'Programming Language :: Python :: 3.11',
            'Topic :: Scientific/Engineering :: Artificial Intelligence',
        ],
        url='https://github.com/innerlee/gcr',
        author='MMPretrain Contributors',
        author_email='inerlee@gmail.com',
        license='Apache License 2.0',
        zip_safe=False)
