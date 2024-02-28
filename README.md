# FaceScan

## Adding Jupyter Notebook to Project Requirements

To ensure that your team members can use Jupyter Notebook when they pull the code, you can add it to your project's requirements. You can achieve this by including it in your `requirements.txt` file, which is commonly used in Python projects to specify dependencies.

### Steps:

1. **Specify Jupyter Notebook Version:**

    Add the following line to your `requirements.txt` file:

    ```plaintext
    jupyter==7.1.1
    ```

    This specifies that version 7.1.1 of Jupyter Notebook is required for the project.

2. **Install Dependencies:**

    When your team members pull the code, they can install all the dependencies listed in `requirements.txt` using `pip`. They should run the following command in their terminal:

    ```bash
    pip install -r requirements.txt
    ```

    This command installs all the packages listed in `requirements.txt`, including Jupyter Notebook, ensuring that they have the necessary dependencies to work with the project.

By following these steps, you can ensure that Jupyter Notebook is included as a dependency for your project, making it easy for your team members to use it when working with the codebase.
