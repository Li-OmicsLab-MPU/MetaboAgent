import { useMemo } from 'react';
import { AgGridReact } from 'ag-grid-react';
import { ColDef } from 'ag-grid-community';
import { Table } from 'lucide-react';

// 导入 AG Grid 样式
import 'ag-grid-community/styles/ag-grid.css';
import 'ag-grid-community/styles/ag-theme-alpine.css';

// ============================================================================
// Mock 数据生成函数
// ============================================================================

const generateMockData = (rows: number, cols: number) => {
  const data = [];
  const groups = ['Control', 'Case'];

  for (let i = 0; i < rows; i++) {
    const row: any = {
      Sample_ID: `Sample_${String(i + 1).padStart(3, '0')}`,
    };

    // 生成代谢物特征列
    for (let j = 1; j <= cols; j++) {
      const metaboliteName = `Metabolite_${j}`;
      // 生成随机浓度值（模拟真实代谢组学数据）
      row[metaboliteName] = (Math.random() * 1000 + 100).toFixed(2);
    }

    // Group 列放在最后
    row.Group = groups[i % 2]; // 交替分配 Control 和 Case

    data.push(row);
  }

  return data;
};

// ============================================================================
// DataTable 组件 - 高性能数据表格
// ============================================================================

export const DataTable = () => {
  // 生成 Mock 数据：50 行 x 20 列
  const rowData = useMemo(() => generateMockData(50, 20), []);

  // 配置列定义
  const columnDefs = useMemo<ColDef[]>(() => {
    if (rowData.length === 0) return [];

    const firstRow = rowData[0];
    const columns: ColDef[] = [];

    // 遍历第一行的所有键来生成列定义
    Object.keys(firstRow).forEach((key) => {
      if (key === 'Sample_ID') {
        // 第一列：Sample_ID - 冻结在左侧
        columns.push({
          field: 'Sample_ID',
          headerName: 'Sample ID',
          pinned: 'left', // 冻结在左侧
          width: 150,
          cellStyle: { fontWeight: 'bold', backgroundColor: '#f9fafb' },
          filter: 'agTextColumnFilter',
          sortable: true,
        });
      } else if (key === 'Group') {
        // 最后一列：Group - 冻结在右侧
        columns.push({
          field: 'Group',
          headerName: 'Group',
          pinned: 'right', // 冻结在右侧
          width: 120,
          cellStyle: (params) => {
            // 根据分组设置不同颜色
            if (params.value === 'Control') {
              return { backgroundColor: '#dbeafe', color: '#1e40af', fontWeight: 'bold' };
            } else if (params.value === 'Case') {
              return { backgroundColor: '#fce7f3', color: '#be185d', fontWeight: 'bold' };
            }
            return null;
          },
          filter: 'agSetColumnFilter',
          sortable: true,
        });
      } else {
        // 中间列：代谢物特征列
        columns.push({
          field: key,
          headerName: key,
          width: 140,
          type: 'numericColumn',
          filter: 'agNumberColumnFilter',
          sortable: true,
          valueFormatter: (params) => {
            // 格式化数值显示
            return params.value ? parseFloat(params.value).toFixed(2) : '';
          },
        });
      }
    });

    return columns;
  }, [rowData]);

  // AG Grid 默认配置
  const defaultColDef = useMemo<ColDef>(() => {
    return {
      resizable: true, // 允许调整列宽
      sortable: true, // 允许排序
      filter: true, // 允许过滤
      floatingFilter: false, // 不显示浮动过滤器（可选）
    };
  }, []);

  return (
    <div className="h-full flex flex-col bg-white rounded-lg shadow">
      {/* ====================================================================
          顶部标题栏
      ==================================================================== */}
      <div className="px-6 py-4 border-b border-gray-200">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Table size={20} className="text-blue-600" />
            <h3 className="text-lg font-semibold text-gray-900">数据表格</h3>
          </div>
          <div className="text-sm text-gray-500">
            {rowData.length} 行 × {columnDefs.length} 列
          </div>
        </div>
        <p className="text-sm text-gray-500 mt-1">
          代谢组学特征数据（Mock 数据）
        </p>
      </div>

      {/* ====================================================================
          AG Grid 表格区域
      ==================================================================== */}
      <div className="flex-1 p-6">
        <div className="ag-theme-alpine h-full w-full">
          <AgGridReact
            rowData={rowData}
            columnDefs={columnDefs}
            defaultColDef={defaultColDef}
            // 性能优化配置
            animateRows={true} // 行动画
            enableCellTextSelection={true} // 允许选择单元格文本
            suppressRowClickSelection={true} // 禁止点击行选择
            // 虚拟滚动（AG Grid 默认启用，无需额外配置）
            // 表头吸顶（AG Grid 默认支持）
            // 其他配置
            pagination={false} // 不使用分页，使用虚拟滚动
            domLayout="normal" // 正常布局模式
            // 主题和样式
            rowHeight={40} // 行高
            headerHeight={48} // 表头高度
          />
        </div>
      </div>

      {/* ====================================================================
          底部统计信息
      ==================================================================== */}
      <div className="px-6 py-4 border-t border-gray-200 bg-gray-50">
        <div className="grid grid-cols-4 gap-4 text-center text-sm">
          <div>
            <div className="text-xs text-gray-600">总样本数</div>
            <div className="font-semibold text-gray-900 mt-1">
              {rowData.length}
            </div>
          </div>
          <div>
            <div className="text-xs text-gray-600">特征数量</div>
            <div className="font-semibold text-gray-900 mt-1">
              {columnDefs.length - 2} {/* 减去 Sample_ID 和 Group */}
            </div>
          </div>
          <div>
            <div className="text-xs text-gray-600">Control 组</div>
            <div className="font-semibold text-blue-600 mt-1">
              {rowData.filter((row) => row.Group === 'Control').length}
            </div>
          </div>
          <div>
            <div className="text-xs text-gray-600">Case 组</div>
            <div className="font-semibold text-pink-600 mt-1">
              {rowData.filter((row) => row.Group === 'Case').length}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
