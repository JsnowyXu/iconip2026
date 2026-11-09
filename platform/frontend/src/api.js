export async function apiGet(path) {
  const res = await fetch(path)
  if (!res.ok) throw new Error(`GET ${path} failed`)
  return await res.json()
}

export async function apiPost(path, payload) {
  const res = await fetch(path, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload || {}),
  })
  const data = await res.json().catch(() => ({}))
  if (!res.ok) throw new Error(data.error || `POST ${path} failed`)
  return data
}
