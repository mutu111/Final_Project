import matplotlib.pyplot as plt
import seaborn as sns


def plot_hist(df, column):
    """
    Plot histogram
    """
    plt.figure(figsize=(8, 5))
    sns.histplot(df[column], kde=True)
    plt.title(f"Distribution of {column}")
    plt.show()


def plot_scatter(df, x, y):
    """
    Plot scatter plot
    """
    plt.figure(figsize=(8, 5))
    sns.regplot(
        data=df,
        x=x,
        y=y,
        scatter_kws={"alpha": 0.5},
        order=2,
        line_kws={"color": "red"},
    )
    sns.scatterplot(data=df, x=x, y=y)
    plt.title(f"{x} vs {y}")
    plt.show()


def plot_correlation(df):
    """
    Plot correlation matrix
    """
    selected_vars = [
        "Violent Rate",
        "Population",
        "Civilian_labor_force",
        "Unemployment_rate",
        "Urban_Influence_Code",
        "Metro",
        "Traffic_Count",
    ]
    corr_df = df[selected_vars].corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_df, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Matrix")
    plt.tight_layout()
    plt.show()


def plot_box(df, x, y):
    """
    Plot boxplot
    """
    plt.figure(figsize=(8, 6))
    sns.boxplot(data=df, x=x, y=y, palette=["skyblue", "salmon"])
    plt.title(f"{y} by {x}")
    plt.xlabel(x)
    plt.ylabel(y)
    plt.tight_layout()
    plt.show()


def plot_box_urban(df, x, y):
    """
    Plot boxplot
    """
    palette = sns.color_palette("Set3", 9)
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df, x=x, y=y, palette=palette)
    plt.title(f"{y} by {x}")
    plt.xlabel(x)
    plt.ylabel(y)
    plt.tight_layout()
    plt.show()
