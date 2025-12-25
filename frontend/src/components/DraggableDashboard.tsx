"use client";

import { useState, useEffect, useCallback } from "react";
import { Responsive, WidthProvider, Layout } from "react-grid-layout";
import "react-grid-layout/css/styles.css";
import "react-resizable/css/styles.css";
import styles from "./DraggableDashboard.module.css";

const ResponsiveGridLayout = WidthProvider(Responsive);

interface Widget {
  id: string;
  title: string;
  icon?: string;
  component: React.ReactNode;
}

interface DraggableDashboardProps {
  widgets: Widget[];
  onLayoutChange?: (layout: Layout[]) => void;
  isEditing?: boolean;
}

const STORAGE_KEY = "kantoorkiller-dashboard-layout-v2";

const defaultLayouts: { [key: string]: Layout[] } = {
  lg: [
    { i: "chat", x: 0, y: 0, w: 4, h: 10, minW: 3, minH: 6 },
    { i: "execution", x: 4, y: 0, w: 5, h: 10, minW: 3, minH: 5 },
    { i: "activity", x: 9, y: 0, w: 3, h: 5, minW: 2, minH: 3 },
    { i: "knowledge", x: 9, y: 5, w: 3, h: 5, minW: 2, minH: 3 },
  ],
  md: [
    { i: "chat", x: 0, y: 0, w: 4, h: 9, minW: 3, minH: 5 },
    { i: "execution", x: 4, y: 0, w: 6, h: 9, minW: 3, minH: 5 },
    { i: "activity", x: 0, y: 9, w: 5, h: 4, minW: 2, minH: 3 },
    { i: "knowledge", x: 5, y: 9, w: 5, h: 4, minW: 2, minH: 3 },
  ],
  sm: [
    { i: "chat", x: 0, y: 0, w: 6, h: 7, minW: 3, minH: 5 },
    { i: "execution", x: 0, y: 7, w: 6, h: 6, minW: 3, minH: 4 },
    { i: "activity", x: 0, y: 13, w: 6, h: 4, minW: 2, minH: 3 },
    { i: "knowledge", x: 0, y: 17, w: 6, h: 4, minW: 2, minH: 3 },
  ],
};

function loadLayouts(): { [key: string]: Layout[] } {
  if (typeof window === "undefined") return defaultLayouts;
  try {
    const saved = localStorage.getItem(STORAGE_KEY);
    if (saved) {
      const parsed = JSON.parse(saved);
      // Validate that all widget IDs exist
      const requiredIds = ["chat", "execution", "activity", "knowledge"];
      const hasAll = requiredIds.every((id) =>
        parsed.lg?.some((l: Layout) => l.i === id)
      );
      if (hasAll) return parsed;
    }
  } catch (e) {
    console.error("Failed to load layout from localStorage:", e);
  }
  return defaultLayouts;
}

export function resetDashboardLayout() {
  if (typeof window === "undefined") return;
  localStorage.setItem(STORAGE_KEY, JSON.stringify(defaultLayouts));
  window.location.reload();
}

function saveLayouts(layouts: { [key: string]: Layout[] }) {
  if (typeof window === "undefined") return;
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(layouts));
  } catch (e) {
    console.error("Failed to save layout to localStorage:", e);
  }
}

export default function DraggableDashboard({ widgets, onLayoutChange, isEditing = false }: DraggableDashboardProps) {
  const [layouts, setLayouts] = useState<{ [key: string]: Layout[] }>(defaultLayouts);
  const [mounted, setMounted] = useState(false);

  useEffect(() => {
    setMounted(true);
    setLayouts(loadLayouts());
  }, []);

  const handleLayoutChange = useCallback(
    (currentLayout: Layout[], allLayouts: { [key: string]: Layout[] }) => {
      setLayouts(allLayouts);
      saveLayouts(allLayouts);
      if (onLayoutChange) {
        onLayoutChange(currentLayout);
      }
    },
    [onLayoutChange]
  );

  if (!mounted) {
    return <div className={styles.loading}>Loading dashboard...</div>;
  }

  return (
    <div className={styles.container}>
      <ResponsiveGridLayout
        className={`${styles.grid} ${isEditing ? styles.editing : ""}`}
        layouts={layouts}
        breakpoints={{ lg: 1200, md: 900, sm: 600 }}
        cols={{ lg: 12, md: 10, sm: 6 }}
        rowHeight={50}
        margin={[12, 12]}
        containerPadding={[0, 0]}
        onLayoutChange={handleLayoutChange}
        isDraggable={isEditing}
        isResizable={isEditing}
        draggableHandle={`.${styles.widgetHeader}`}
        useCSSTransforms={true}
        compactType="vertical"
        preventCollision={false}
      >
        {widgets.map((widget) => (
          <div key={widget.id} className={styles.widget}>
            <div className={styles.widgetHeader}>
              <span className={styles.widgetTitle}>{widget.title}</span>
              {isEditing && <span className={styles.dragHandle}>::</span>}
            </div>
            <div className={styles.widgetContent}>{widget.component}</div>
          </div>
        ))}
      </ResponsiveGridLayout>
    </div>
  );
}
