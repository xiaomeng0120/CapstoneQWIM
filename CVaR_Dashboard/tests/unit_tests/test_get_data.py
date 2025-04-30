# test_get_data.py
import pytest
import pandas as pd
from unittest.mock import patch, MagicMock
from pathlib import Path
import numpy as np

# 1. 增强的目录隔离配置（解决文件路径问题）
@pytest.fixture(scope="function")
def test_dirs(tmp_path):
    """创建带时间戳的独立测试目录"""
    test_root = tmp_path / f"test_{pd.Timestamp.now().value}"
    raw_dir = test_root / "raw"
    processed_dir = test_root / "processed"
    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(exist_ok=True)
    return {
        "raw": raw_dir,
        "processed": processed_dir
    }

# 2. 强化模拟数据生成（解决EmptyDataError）
@pytest.fixture
def mock_etf_data():
    """生成符合yfinance格式的ETF数据"""
    dates = pd.date_range("2023-01-01", periods=5)
    return pd.DataFrame(
        {"SPY": [380.1, 382.5, 381.9, 383.4, 384.0]},
        index=pd.DatetimeIndex(dates, name='Date')
    )

@pytest.fixture
def valid_ff5_file(tmp_path):
    """生成标准FF5数据文件（包含列名和分隔符）"""
    content = """Date,Mkt-RF,SMB,HML,RMW,CMA,RF
2023-01-01,0.52,0.13,-0.21,0.08,-0.04,0.001
2023-01-02,-0.15,0.07,0.12,0.05,-0.02,0.001"""
    path = tmp_path / "ff5.csv"
    path.write_text(content)
    return path

@pytest.fixture 
def valid_mom_file(tmp_path):
    """生成标准Momentum数据文件"""
    content = """Date,MOM
2023-01-01,1.23
2023-01-02,-0.45"""
    path = tmp_path / "mom.csv"
    path.write_text(content)
    return path

# 3. 修复ETF结构测试
def test_etf_structure(test_dirs, mock_etf_data):
    """验证数据结构及保存格式"""
    with patch("yfinance.download") as mock_dl:
        mock_dl.return_value = MagicMock(Close=mock_etf_data)
        
        from src.utils.get_data import get_etf_data
        save_path = test_dirs["raw"]/"etf_test.csv"
        
        # 显式检查路径存在性
        assert save_path.parent.exists(), "目录未正确创建"
        
        df = get_etf_data(
            tickers=["SPY"],
            start_date="2023-01-01",
            end_date="2023-01-05",
            save_path=save_path
        )
        
        # 增强文件存在性检查
        assert save_path.exists(), "文件未成功保存"
        
        # 验证保存格式
        saved = pd.read_csv(save_path, 
                          parse_dates=['Date'],
                          index_col='Date')
        assert isinstance(saved.index, pd.DatetimeIndex)
        assert saved.index.name == "Date"
        assert 'SPY' in saved.columns

# 4. 修复FF5+MOM处理测试
def test_ff5_mom_processing(test_dirs, valid_ff5_file, valid_mom_file):
    """验证因子数据处理流程"""
    from src.utils.get_data import get_ff5_mom_data
    
    output_path = test_dirs["processed"]/"ff5_mom.csv"
    
    # 强制路径存在检查
    assert valid_ff5_file.exists() and valid_mom_file.exists()
    
    df = get_ff5_mom_data(
        ff5_file=str(valid_ff5_file),
        mom_file=str(valid_mom_file),
        start_date="2023-01-01",
        end_date="2023-01-02",
        save_path=output_path
    )
    
    # 验证合并结果
    assert df.shape == (2, 7), f"实际维度: {df.shape}"
    assert list(df.columns) == ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "RF", "MOM"]
    assert pd.api.types.is_datetime64_any_dtype(df.index)

# 5. 修复日期过滤测试
def test_date_filtering(valid_ff5_file, valid_mom_file):
    """验证日期范围过滤逻辑"""
    from src.utils.get_data import get_ff5_mom_data
    
    df = get_ff5_mom_data(
        ff5_file=str(valid_ff5_file),
        mom_file=str(valid_mom_file),
        start_date="2023-01-02",
        end_date="2023-01-02",
        save_path="dummy.csv"
    )
    
    assert df.shape[0] == 1, f"实际行数: {df.shape[0]}"
    assert df.index[0] == pd.Timestamp('2023-01-02')

# 6. 强化非法日期测试
def test_invalid_dates(test_dirs):
    """精确匹配异常消息"""
    with patch("yfinance.download") as mock_dl:
        mock_dl.return_value = MagicMock(Close=pd.DataFrame())
        
        from src.utils.get_data import get_etf_data
        
        # 使用正则表达式精确匹配
        error_pattern = r".*YYYY-MM-DD.*"
        
        with pytest.raises(
            ValueError,
            match=error_pattern
        ):
            get_etf_data(
                tickers=["SPY"],
                start_date="2023/13/01",  # 错误的分隔符和月份
                end_date="2023-01-32",    # 非法日期
                save_path=test_dirs["raw"]/"invalid.csv"
            )

# 7. 空文件处理测试
def test_empty_data(test_dirs):
    """验证空文件异常处理"""
    from src.utils.get_data import get_ff5_mom_data
    
    empty_path = test_dirs["raw"]/"empty.csv"
    empty_path.touch()  # 创建空文件
    
    with pytest.raises(
        pd.errors.EmptyDataError,
        match="No columns to parse from file"
    ):
        get_ff5_mom_data(
            ff5_file=str(empty_path),
            mom_file=str(empty_path),
            start_date="2023-01-01",
            end_date="2023-01-05",
            save_path=empty_path
        )