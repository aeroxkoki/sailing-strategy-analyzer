/**
 * APIデータフェッチングフック
 */
import useSWR, { SWRConfiguration, SWRResponse } from 'swr';
import apiClient, { ApiResponse, ApiError } from '../lib/api';

/**
 * フック戻り値の型
 */
export interface UseApiResponse<T> extends Omit<SWRResponse<ApiResponse<T>, ApiError>, 'data'> {
  data: T | undefined;
  isLoading: boolean;
  isError: boolean;
  response: ApiResponse<T> | undefined;
}

/**
 * APIデータを取得するためのSWRフック
 * 
 * @param url APIのURL
 * @param config SWRの設定オプション
 * @returns データ、ロード状態、エラー状態などを含むオブジェクト
 */
export function useApi<T>(url: string | null, config?: SWRConfiguration): UseApiResponse<T> {
  // null URLの場合はデータをフェッチしない
  const shouldFetch = !!url;
  
  const fetcher = async (path: string) => {
    try {
      return await apiClient.get<T>(path);
    } catch (error) {
      throw error;
    }
  };

  const { data, error, mutate, isValidating, ...rest } = useSWR<ApiResponse<T>, ApiError>(
    shouldFetch ? url : null,
    fetcher,
    {
      revalidateOnFocus: false,
      revalidateOnReconnect: true,
      errorRetryCount: 3,
      ...config,
    }
  );

  return {
    data: data?.data,
    response: data,
    isLoading: shouldFetch && !error && !data,
    isError: !!error,
    error,
    mutate,
    isValidating,
    ...rest,
  };
}

/**
 * 投稿リクエスト用フック
 */
export const usePost = <T, P = any>() => {
  const post = async (url: string, payload: P): Promise<ApiResponse<T>> => {
    try {
      return await apiClient.post<T>(url, payload);
    } catch (error) {
      throw error;
    }
  };

  return { post };
};

/**
 * 更新リクエスト用フック
 */
export const usePut = <T, P = any>() => {
  const put = async (url: string, payload: P): Promise<ApiResponse<T>> => {
    try {
      return await apiClient.put<T>(url, payload);
    } catch (error) {
      throw error;
    }
  };

  return { put };
};

/**
 * 削除リクエスト用フック
 */
export const useDelete = <T>() => {
  const del = async (url: string): Promise<ApiResponse<T>> => {
    try {
      return await apiClient.delete<T>(url);
    } catch (error) {
      throw error;
    }
  };

  return { del };
};

export default useApi;
