use clap::Parser;

#[derive(Parser)]
#[command(author, version)]
pub(crate) struct Cli {
    /// 模型路径
    #[arg(short, long)]
    pub(crate) model_path: String,
    /// 模式
    #[command(subcommand)]
    pub(crate) mode: Mode,
}

#[derive(Parser)]
pub(crate) enum Mode {
    /// 单次生成模式
    Once(OnceMode),
    /// 聊天模式
    Chat(ChatMode),
    /// Web模式
    Web(WebMode),
}

#[derive(Parser)]
pub(crate) struct ModelArgs {
    /// 输出长度限制
    #[clap(short, long, default_value_t = 512)]
    pub(crate) max_length: u32,
    /// Top-k 输出
    #[clap(short = 'k', long, default_value_t = 10)]
    pub(crate) top_k: u32,
    /// Temperature 控制
    #[clap(short, long, default_value_t = 1.0)]
    pub(crate) temperature: f32,
    /// Top-p 控制
    #[clap(short = 'p', long, default_value_t = 0.9)]
    pub(crate) top_p: f32,
}

#[derive(Parser)]
pub(crate) struct OnceMode {
    /// 输入提示词
    #[clap(short, long)]
    pub(crate) prompt: String,
    #[command(flatten)]
    pub(crate) model_args: ModelArgs,
}

#[derive(Parser)]
pub(crate) struct ChatMode {
    /// 系统提示
    #[clap(short, long)]
    pub(crate) system_prompt: Option<String>,
    #[command(flatten)]
    pub(crate) model_args: ModelArgs,
}

#[derive(Parser)]
pub(crate) struct WebMode {
    /// 端口号
    #[clap(long, default_value_t = 8000)]
    pub(crate) port: u16,
    /// host
    #[clap(long, default_value = "127.0.0.1")]
    pub(crate) host: String,
    #[command(flatten)]
    pub(crate) model_args: ModelArgs,
}
