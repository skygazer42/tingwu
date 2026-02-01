import { Outlet } from 'react-router-dom'
import { Sidebar } from './Sidebar'
import { Navbar } from './Navbar'
import { ScrollArea } from '@/components/ui/scroll-area'

export function AppLayout() {
  return (
    <div className="flex h-screen bg-background">
      {/* 侧边栏 */}
      <Sidebar />

      {/* 主内容区 */}
      <div className="flex flex-1 flex-col overflow-hidden">
        {/* 顶部导航 */}
        <Navbar />

        {/* 页面内容 */}
        <ScrollArea className="flex-1">
          <main className="container mx-auto py-6 px-4 md:px-6">
            <Outlet />
          </main>
        </ScrollArea>
      </div>
    </div>
  )
}
