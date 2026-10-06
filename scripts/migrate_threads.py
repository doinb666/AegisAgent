"""先预览，再显式备份迁移；不输出数据库凭据。"""

from app.harness.thread_migration import main

if __name__ == "__main__":
    main()
