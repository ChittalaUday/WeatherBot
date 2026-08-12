"use client";

import {
  flexRender,
  getCoreRowModel,
  getSortedRowModel,
  useReactTable,
  type ColumnDef,
  type SortingState,
} from "@tanstack/react-table";
import { ArrowUpDown } from "lucide-react";
import { useMemo, useState } from "react";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import type { TableData } from "@/lib/types";

/** Sortable forecast table. Columns are whatever the intent mapped to, so they are built at runtime. */
export function ResultTable({ data }: { data: TableData }) {
  const [sorting, setSorting] = useState<SortingState>([]);

  const columns = useMemo<ColumnDef<Record<string, string>>[]>(
    () =>
      data.columns.map((column) => ({
        accessorKey: column.key,
        header: ({ column: col }) => (
          <button
            className="inline-flex items-center gap-1 text-left font-medium hover:text-foreground"
            onClick={() => col.toggleSorting(col.getIsSorted() === "asc")}
          >
            {column.label}
            <ArrowUpDown className="h-3 w-3 opacity-40 transition-opacity group-hover:opacity-100" />
          </button>
        ),
        cell: (info) => <span className="tabular-nums">{String(info.getValue() ?? "-")}</span>,
      })),
    [data.columns],
  );

  const table = useReactTable({
    data: data.rows,
    columns,
    state: { sorting },
    onSortingChange: setSorting,
    getCoreRowModel: getCoreRowModel(),
    getSortedRowModel: getSortedRowModel(),
  });

  return (
    <div className="mt-3 overflow-x-auto rounded-lg border bg-background/60">
      <Table>
        <TableHeader>
          {table.getHeaderGroups().map((group) => (
            <TableRow key={group.id} className="hover:bg-transparent">
              {group.headers.map((header) => (
                <TableHead key={header.id} className="group whitespace-nowrap text-xs">
                  {header.isPlaceholder ? null : flexRender(header.column.columnDef.header, header.getContext())}
                </TableHead>
              ))}
            </TableRow>
          ))}
        </TableHeader>
        <TableBody>
          {table.getRowModel().rows.map((row) => (
            <TableRow key={row.id} className="text-sm">
              {row.getVisibleCells().map((cell) => (
                <TableCell key={cell.id} className="whitespace-nowrap py-2">
                  {flexRender(cell.column.columnDef.cell, cell.getContext())}
                </TableCell>
              ))}
            </TableRow>
          ))}
        </TableBody>
      </Table>
    </div>
  );
}
